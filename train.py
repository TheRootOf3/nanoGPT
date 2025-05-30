"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import (
    GPTConfig,
    GPT,
    CausalSelfAttention,
    SplitCausalSelfAttentionVariableNumHeadsIndependent,
)
from attention_head_similarity import (
    cosine_similarity,
    cka,
    compute_pairwise_symmetric_similarity_matrix,
    compute_aggr_pairwise_similarity,
    compute_mean_per_head_similarities,
    compute_max_per_head_similarities,
)
from matplotlib import pyplot as plt

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
block_size = 1024
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
output_attentions = False  # whether to output attention weights for each block
override_checkpoint = True

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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
print(f"max num of iterations will be: {int(max_iters)}")
print(f"num of warmup iterations will be: {int(warmup_iters)}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
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
    block_size=block_size,
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
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
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
    n_blocks = gptconf.n_layer  # number of blocks in the model
    cka_results = []
    cossim_results = []
    head_idx = []
    # we will use the cosine similarity metric for the CKA
    with ctx:
        _, _, attn_weights = model(X_sim, Y_sim, output_attentions=True)
    if attn_weights[0].shape[1] == 1:
        return None
    for block_idx in range(n_blocks):
        S_cka = compute_pairwise_symmetric_similarity_matrix(
            attn_weights[block_idx], cka
        )
        S_cossim = compute_pairwise_symmetric_similarity_matrix(
            attn_weights[block_idx], cosine_similarity
        )
        cka_results.append(S_cka)
        cossim_results.append(S_cossim)

        local_model = model if not ddp else model.module
        if isinstance(
            local_model.transformer.h[block_idx].attn,
            SplitCausalSelfAttentionVariableNumHeadsIndependent,
        ):
            head_idx.append(local_model.transformer.h[block_idx].attn.trainable_heads)

    out["cka_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.mean) for S in cka_results
    ]
    out["cossim_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.mean) for S in cossim_results
    ]
    out["cka_max_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.max) for S in cka_results
    ]
    out["cossim_max_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.max) for S in cka_results
    ]
    out["cka_std_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.std) for S in cka_results
    ]
    out["cossim_std_per_block"] = [
        compute_aggr_pairwise_similarity(S, torch.std) for S in cossim_results
    ]
    out["cka"] = cka_results
    out["cossim"] = cossim_results
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def log_similarity_heatmap(S: torch.Tensor, step: int, name: str):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(S, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"Head-Head Similarity\n{name}, step: {step}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Head index")

    # Add ticks between 0 and S.shape[0], centered
    tick_positions = np.arange(S.shape[0])
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)
    ax.tick_params(axis="both", which="major", labelsize=8)

    # Add value annotations
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            value = S[i, j].item()
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if value < 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Log the figure
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_per_head_model_heatmap(
    per_layer_head_similarities: list[torch.Tensor],
    step: int,
    name: str,
):
    max_n_heads = max([len(s) for s in per_layer_head_similarities])
    # Ensure all tensors have the same number of heads by right padding with zeros
    per_layer_head_similarities = [
        torch.cat(
            [
                s,
                torch.zeros(max_n_heads - len(s)),
            ]
        )
        for s in per_layer_head_similarities
    ]

    S = torch.vstack(per_layer_head_similarities[::-1])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(S, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(f"Per-Head Model CKA Similarity Heatmap, step: {step}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")

    # Set y-ticks to reflect the correct layer indices (reversed)
    num_layers = S.shape[0]
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers - 1, -1, -1))  # Reversed layer indices

    # Set x-ticks to reflect the correct head indices
    num_heads = S.shape[1]
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels(range(num_heads))
    ax.tick_params(axis="both", which="major", labelsize=12)
    # Add value annotations
    for i in range(S.shape[0]):  # iterate over layers
        for j in range(S.shape[1]):
            value = S[i, j].item()
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value < 0.5 else "black",
            )

    # Add colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Log the figure
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close(fig)


def get_head_max_similarity_threshold(iter_num: int) -> float:
    # cosine schedule for head max simiality threshold

    # Max similarity threshold means that we will not add any new heads until there exists pair of heads
    # with similarity value lower than the threshold. This promotes the model to learn different heads and
    # gradually allows for more similarity as the training progresses.

    t_0 = 0.4  # initial threshold
    t_1 = 0.9  # final threshold
    t = iter_num / max_iters
    return t_0 + (t_1 - t_0) * (1 - math.cos(math.pi * t)) / 2


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log_freq=log_interval)

X_sim, Y_sim = get_batch("val")  # fetch the very first batch for similarity

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
    True  # whether to copy the heads from the previous layer when adding new heads
)
while True:

    local_model = model.module if ddp else model  # unwrap DDP container if needed
    if attention_layer == "selective":
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
    lr = get_lr(iter_num) if decay_lr else learning_rate
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
                        f"cka/block_{i}": attn["cka_per_block"][i]
                        for i in range(len(attn["cka_per_block"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/block_{i}": attn["cossim_per_block"][i]
                        for i in range(len(attn["cossim_per_block"]))
                    }
                )
                log_data.update(
                    {
                        f"cka/std_block_{i}": attn["cka_std_per_block"][i]
                        for i in range(len(attn["cka_std_per_block"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/std_block_{i}": attn["cossim_std_per_block"][i]
                        for i in range(len(attn["cossim_std_per_block"]))
                    }
                )
                log_data.update(
                    {
                        f"cka/max_block_{i}": attn["cka_max_per_block"][i]
                        for i in range(len(attn["cka_max_per_block"]))
                    }
                )
                log_data.update(
                    {
                        f"cossim/max_block_{i}": attn["cossim_max_per_block"][i]
                        for i in range(len(attn["cossim_max_per_block"]))
                    }
                )

                for block_id in range(len(attn["cka"])):
                    log_similarity_heatmap(
                        attn["cka"][block_id],
                        step=iter_num,
                        name=f"cka_heatmap/block_{block_id}",
                    )
                    log_similarity_heatmap(
                        attn["cossim"][block_id],
                        step=iter_num,
                        name=f"cossim_heatmap/block_{block_id}",
                    )

                    head_sim_th = get_head_max_similarity_threshold(iter_num)
                    if attn["cka_max_per_block"][block_id] < head_sim_th:
                        add_new_heads[block_id] = True

                mean_per_head_sims = [
                    compute_mean_per_head_similarities(attn["cka"][i])
                    for i in range(len(attn["cka"]))
                ]

                max_per_head_sims = [
                    compute_max_per_head_similarities(attn["cka"][i])
                    for i in range(len(attn["cka"]))
                ]

                log_per_head_model_heatmap(
                    mean_per_head_sims, iter_num, "cka_per_head/mean_model_head_heatmap"
                )
                log_per_head_model_heatmap(
                    max_per_head_sims, iter_num, "cka_per_head/max_model_head_heatmap"
                )

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

    t0 = time.time()
    # Add the attention head splitting
    # The idea is that we only backprop through and train only one half of the attention heads
    # and in the next iteration we train the other half

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
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
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
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
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "iter_time": dt * 1000,
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                    "head_max_similarity_threshold": get_head_max_similarity_threshold(
                        iter_num
                    ),
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
