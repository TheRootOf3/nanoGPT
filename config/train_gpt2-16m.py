# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


wandb_log = True
wandb_project = "owt"
wandb_run_name = "gpt2-16M-split-2-of-8-nanogpt-1"
# wandb_run_name = "gpt2-16M-causal"

out_dir = "gpt2-16M-split-2-of-8-nanogpt-1"

# model - 16M GPT-2
max_seq_length = 512
n_layer = 4
n_head = 8
n_embd = 256
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = True  # do we use bias inside LayerNorm and Linear layers?


batch_size = 32
gradient_accumulation_steps = 5 * 1


# this makes total number of tokens be ~327M
max_iters = 4_000
lr_decay_iters = 4_000

# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
compile = False

# attention_layer = "causal"
attention_layer = "selective"
modify_number_of_heads = True

save_checkpoint = False
override_checkpoint = False

learning_rate = 1e-3  # max learning rate
warmup_iters = 400
decay_lr = True

scheduler_type = "wsd"

SEED = 1337
