#!/bin/bash
#SBATCH --job-name=hipmri_unet
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/Student/s4882492/PatternAnalysis-2025/recognition/hipmri_unet_plus/slurm_%j.out
#SBATCH --chdir=/home/Student/s4882492/PatternAnalysis-2025/recognition/hipmri_unet_plus

source ~/PatternAnalysis-2025/.venv/bin/activate
cd ~/PatternAnalysis-2025/recognition/hipmri_unet_plus

python train.py \
  --images_dir /home/groups/comp3710/HipMRI_Study_open/semantic_MRs \
  --labels_dir /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only \
  --epochs 40 --batch_size 8 --lr 1e-3 \
  --out_dir runs/hipmri_v1 \
  --num_classes 6 --val_split 0.15 --test_split 0.15 \
  --num_workers 4
