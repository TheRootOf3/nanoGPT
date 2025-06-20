#!/bin/bash
#SBATCH -w ruapehu
#SBATCH --cpus-per-task 40
#SBATCH --gres=gpu:2
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --job-name=train-gpt2

hostname
date

export PYTHONUNBUFFERED=TRUE

# poetry run torchrun --nproc_per_node 2 --master_port 29501 train.py config/train_gpt2.py

# sleep 5

poetry run torchrun --nproc_per_node 2 --master_port 29501 train.py config/train_gpt2.py --scheduler_type=cosine --wandb_run_name=gpt2-124M-split-full-cosine --out_dir=out-124M-split-full-cosine

date
