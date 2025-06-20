# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "owt"
wandb_run_name = "gpt2-124M"

# model - 16M GPT-2
max_seq_length = 8192
n_layer = 16
n_head = 32
n_embd = 2048
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?


batch_size = 1
gradient_accumulation_steps = 5 * 4


# this makes total number of tokens be ~327M
max_iters = 2_000
lr_decay_iters = 2_000

# eval stuff
eval_interval = 100
eval_iters = 5
log_interval = 10

# weight decay
weight_decay = 1e-1
compile = False

attention_layer = "causal"
# attention_layer = "selective"

learning_rate = 6e-4  # max learning rate
