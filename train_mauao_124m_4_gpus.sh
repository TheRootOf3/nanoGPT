#!/bin/bash
#SBATCH -w mauao
#SBATCH --cpus-per-task 40
#SBATCH --gres=gpu:4
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --time=08:00:00
#SBATCH --job-name=train-gpt2

hostname
date

export PYTHONUNBUFFERED=TRUE

poetry run torchrun --nproc_per_node 4 --master_port 29501 train.py config/train_gpt2.py \
    --wandb_run_name=gpt2-124M-split-scheduled-nanogpt \
    --out_dir=gpt2-124M-split-scheduled-nanogpt \
    --SEED=1958

date
