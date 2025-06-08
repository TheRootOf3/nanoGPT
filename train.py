import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from matplotlib import pyplot as plt

from model import (
    GPTConfig,
    GPT,
    CausalSelfAttention,
    SplitCausalSelfAttentionVariableNumHeadsIndependent,
)

from plotting.similarity_heatmaps import (
    get_per_head_model_heatmap,
    get_similarity_heatmap,
)

from schedulers import head_max_similarity_schedule, wsd_schedule

from attention_head_similarity import (
    cosine_similarity,
    cka,
    compute_pairwise_symmetric_similarity_matrix,
    compute_aggr_pairwise_similarity,
    compute_mean_per_head_redundancy,
    compute_max_per_head_redundancy,
    biased_hsic,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        # "font.sans-serif": "Helvetica",
    }
)


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
# model - 124M GPT-2
d_model = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
attention_layer = "causal"  # 'selective'
output_attentions = False  # whether to output attention weights for each layer
override_checkpoint = True
modify_number_of_heads = False

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file

attention_layer_type = (
    CausalSelfAttention
    if attention_layer == "causal"
    else (
        SplitCausalSelfAttentionVariableNumHeadsIndependent
        if attention_layer == "selective"
        else None
    )
)
assert (
    attention_layer_type is not None
), f"unknown attention layer type: {attention_layer}"

config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * d_model
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"max num of iterations will be: {int(max_iters)}")
print(f"num of warmup iterations will be: {int(warmup_iters)}")

if master_process:
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - d_model, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + d_model]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + d_model]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=d_model,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    attention_layer=attention_layer_type,
    output_attentions=output_attentions,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if d_model < model.config.block_size:
    model.crop_block_size(d_model)
    model_args["block_size"] = (
        d_model  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_attention_head_similarity(X_sim, Y_sim):
    out = {}
    model.eval()
    n_layer = gptconf.n_layer  # number of layers in the model
    cka_results = []
    cossim_results = []
    head_idx = []
    # we will use the cosine similarity metric for the CKA
    with ctx:
        _, _, attn_weights = model(X_sim, Y_sim, output_attentions=True)
    if attn_weights[0].shape[1] == 1:
        return None
    for layer_idx in range(n_layer):
        S_cka = compute_pairwise_symmetric_similarity_matrix(
            attn_weights[layer_idx], cka
        )
        S_cossim = compute_pairwise_symmetric_similarity_matrix(
            attn_weights[layer_idx], cosine_similarity
        )
        cka_results.append(S_cka)
        cossim_results.append(S_cossim)

        local_model = model if not ddp else model.module
        if isinstance(
            local_model.transformer.h[layer_idx].attn,
            SplitCausalSelfAttentionVariableNumHeadsIndependent,
        ):
            head_idx.append(local_model.transformer.h[layer_idx].attn.trainable_heads)

    out["cka_mean_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.mean) for S in cka_results
    ]
    out["cossim_mean_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.mean) for S in cossim_results
    ]
    out["cka_max_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.max) for S in cka_results
    ]
    out["cossim_max_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.max) for S in cka_results
    ]
    out["cka_std_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.std) for S in cka_results
    ]
    out["cossim_std_redundancy_per_layer"] = [
        compute_aggr_pairwise_similarity(S, torch.std) for S in cossim_results
    ]
    out["cka"] = cka_results
    out["cossim"] = cossim_results
    model.train()
    return out


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log_freq=log_interval)

X_sim, Y_sim = (
    torch.cat(x, dim=0) for x in zip(*[get_batch("val") for _ in range(5)])
)  # fetch 5 batches for similarity estimation


# training loop
X, Y = get_batch("train")  # fetch the very first batch
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
add_new_heads = [False for _ in range(gptconf.n_layer)]
last_time_heads_added = [
    0 for _ in range(gptconf.n_layer)
]  # when was the last time we added new heads
last_time_heads_added_fraction = 0.1
copy_heads = (
    False  # whether to copy the heads from the previous layer when adding new heads
)
print(model)
lr_scheduler = wsd_schedule(
    n_iterations=max_iters,
    final_lr_factor=0.1,
    fract_warmup=0.1,
    init_div_factor=100,
    fract_decay=0.2,
    decay_type="sqrt",
)
attn_head_sim_scheduler = head_max_similarity_schedule(max_iters, 0.3, 0.9)
while True:

    local_model = model.module if ddp else model  # unwrap DDP container if needed
    if attention_layer == "selective" and modify_number_of_heads:
        for i, layer in enumerate(local_model.transformer.h):
            # check if we need to add new heads in this layer
            if (
                add_new_heads[i]
                and (iter_num - last_time_heads_added[i]) / max_iters
                > last_time_heads_added_fraction
            ):
                if len(layer.attn.trainable_heads) <= gptconf.n_head // 2:
                    # add new heads to the model
                    print(
                        f"step {iter_num}: adding new heads to layer {i}, "
                        f"last time added was {last_time_heads_added[i]}"
                    )
                    # double the number of trainable heads
                    current_n_heads = len(layer.attn.trainable_heads)
                    if copy_heads:
                        layer.attn.copy_heads(
                            layer.attn.trainable_heads,
                            list(range(current_n_heads, 2 * current_n_heads)),
                        )

                    layer.attn.set_trainable_heads(list(range(2 * current_n_heads)))
                    last_time_heads_added[i] = iter_num
                # turn off the flag to add new heads even when not added but can't add anymore
                add_new_heads[i] = False

    # determine and set the learning rate for this iteration
    lr = learning_rate * lr_scheduler(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        attn = estimate_attention_head_similarity(X_sim, Y_sim)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            log_data = {
                "train/loss_est": losses["train"],
                "val/loss_est": losses["val"],
            }
            if attn is not None:
                log_data.update(
                    {
                        f"cka/layer_{i}": attn["cka_mean_redundancy_per_layer"][i]
                        for i in range(len(attn["cka_mean_redundancy_per_layer"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/layer_{i}": attn["cossim_mean_redundancy_per_layer"][i]
                        for i in range(len(attn["cossim_mean_redundancy_per_layer"]))
                    }
                )
                log_data.update(
                    {
                        f"cka/std_layer_{i}": attn["cka_std_redundancy_per_layer"][i]
                        for i in range(len(attn["cka_std_redundancy_per_layer"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/std_layer_{i}": attn[
                            "cossim_std_redundancy_per_layer"
                        ][i]
                        for i in range(len(attn["cossim_std_redundancy_per_layer"]))
                    }
                )
                log_data.update(
                    {
                        f"cka/max_layer_{i}": attn["cka_max_redundancy_per_layer"][i]
                        for i in range(len(attn["cka_max_redundancy_per_layer"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/max_layer_{i}": attn[
                            "cossim_max_redundancy_per_layer"
                        ][i]
                        for i in range(len(attn["cossim_max_redundancy_per_layer"]))
                    }
                )

                for layer_id in range(len(attn["cka"])):
                    name = f"cka_similarity_S/layer_{layer_id}"
                    fig = get_similarity_heatmap(
                        attn["cka"][layer_id],
                        step=iter_num,
                        name=name,
                        plot_title=r"Pairwise CKA Similarity $\mathbf{S}$"
                        + f"\nLayer: {layer_id}, step: {iter_num}",
                        out_dir=out_dir,
                    )
                    wandb.log({name: wandb.Image(fig)}, step=iter_num)
                    plt.close(fig)

                    name = f"cossim_similarity_S/layer_{layer_id}"
                    fig = get_similarity_heatmap(
                        attn["cossim"][layer_id],
                        step=iter_num,
                        name=name,
                        plot_title=r"Pairwise Cosine Similarity $\mathbf{S}$"
                        + f"\nLayer: {layer_id}, step: {iter_num}",
                        out_dir=out_dir,
                    )
                    wandb.log({name: wandb.Image(fig)}, step=iter_num)
                    plt.close(fig)

                    head_sim_th = attn_head_sim_scheduler(iter_num)
                    if attn["cka_max_redundancy_per_layer"][layer_id] < head_sim_th:
                        add_new_heads[layer_id] = True

                    # compute similarity in the weight space
                    with torch.no_grad():
                        _attn = local_model.transformer.h[layer_id].attn
                        Wqks = []
                        for i in _attn.trainable_heads:
                            Wk = _attn.c_attn.weight[
                                0 * _attn.n_embd
                                + i * _attn.head_dim : 0 * _attn.n_embd
                                + (i + 1) * _attn.head_dim,
                                :,
                            ]
                            Wq = _attn.c_attn.weight[
                                1 * _attn.n_embd
                                + i * _attn.head_dim : 1 * _attn.n_embd
                                + (i + 1) * _attn.head_dim,
                                :,
                            ]
                            Wqks.append(Wq @ Wk.T)
                        stacked_Wqk = torch.stack(Wqks, dim=0).unsqueeze(
                            0
                        )  # shape (1, n_heads, head_dim, head_dim)

                        S_cka_wqk = compute_pairwise_symmetric_similarity_matrix(
                            stacked_Wqk, lambda x, y: cka(x, y, biased_hsic)
                        )
                        name = f"cka_heatmap_for_weights/layer_{layer_id}"
                        fig = get_similarity_heatmap(
                            S_cka_wqk,
                            step=iter_num,
                            name=name,
                            plot_title=r"Pairwise CKA Similarity $\mathbf{QK}^T$"
                            + f"\nLayer: {layer_id}, step: {iter_num}",
                            out_dir=out_dir,
                        )
                        wandb.log({name: wandb.Image(fig)}, step=iter_num)
                        plt.close(fig)

                mean_per_head_sims = [
                    compute_mean_per_head_redundancy(attn["cka"][i])
                    for i in range(len(attn["cka"]))
                ]

                max_per_head_sims = [
                    compute_max_per_head_redundancy(attn["cka"][i])
                    for i in range(len(attn["cka"]))
                ]

                name = "cka_per_head/mean_head_redundancy_heatmap"
                fig = get_per_head_model_heatmap(
                    mean_per_head_sims,
                    iter_num,
                    name,
                    out_dir,
                    plot_title=f"Mean Head Redundancy Heatmap\nLayer: {layer_id}, step: {iter_num}",
                )
                wandb.log({name: wandb.Image(fig)}, step=iter_num)
                plt.close(fig)

                name = "cka_per_head/max_head_redundancy_heatmap"
                fig = get_per_head_model_heatmap(
                    max_per_head_sims,
                    iter_num,
                    name,
                    out_dir,
                    plot_title=f"Max Head Redundancy Heatmap\nLayer: {layer_id}, step: {iter_num}",
                )
                wandb.log({name: wandb.Image(fig)}, step=iter_num)
                plt.close(fig)
                name = None

            wandb.log(log_data, step=iter_num)

        if save_checkpoint:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": raw_model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            name = "ckpt.pt" if override_checkpoint else f"ckpt_{iter_num}.pt"
            torch.save(checkpoint, os.path.join(out_dir, name))
    if iter_num == 0 and eval_only:
        break

    t_forward = 0
    t_backward = 0
    t0 = time.time()
    # Add the attention head splitting
    # The idea is that we only backprop through and train only one half of the attention heads
    # and in the next iteration we train the other half

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        t_forward_0 = time.time()
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            t_forward += time.time() - t_forward_0

            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        t_backward_0 = time.time()
        scaler.scale(loss).backward()
        t_backward += time.time() - t_backward_0
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, time forward {((t_forward)*1000):.2f}ms, time backward {((t_backward)*1000):.2f}ms"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "iter_time": dt * 1000,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "head_max_similarity_threshold": attn_head_sim_scheduler(iter_num),
                },
                step=iter_num,
            )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
