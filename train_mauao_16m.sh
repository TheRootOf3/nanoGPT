#!/bin/bash
#SBATCH -w mauao
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --job-name=train-gpt2

hostname
date

export PYTHONUNBUFFERED=TRUE

SEED=1958

poetry run python3 train.py config/train_gpt2-16m.py \
    --SEED=$SEED \
    --wandb_run_name=gpt2-16M-scheduled-copy-nanogpt-$SEED \
    --out_dir=gpt2-16M-split-scheduled-copy-nanogpt-$SEED

date
