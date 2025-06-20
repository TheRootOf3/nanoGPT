# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


wandb_log = True
wandb_project = "owt"
wandb_run_name = "gpt2-124M-split-full"

out_dir = "out-124M-split-full"


batch_size = 16
max_seq_length = 1024
gradient_accumulation_steps = 5 * 4

# this makes total number of tokens be 3B with 4 GPUs
max_iters = 10_000
lr_decay_iters = 10_000
warmup_iters = 500

# eval stuff
eval_interval = 200
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
compile = False

# attention_layer = "causal"
attention_layer = "selective"
modify_number_of_heads = False

learning_rate = 6e-4  # max learning rate
save_checkpoint = False
override_checkpoint = False
decay_lr = True

scheduler_type = "wsd"

SEED = 1958
