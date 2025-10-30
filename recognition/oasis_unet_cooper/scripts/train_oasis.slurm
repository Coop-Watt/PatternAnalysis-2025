#!/bin/bash
#SBATCH --job-name=oasis_unet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out

set -e

module load python/3.10 || true
source .venv/bin/activate || python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

python train.py   --images_dir /home/groups/comp3710/OASIS/images   --labels_dir /home/groups/comp3710/OASIS/labels   --epochs 60 --batch_size 8 --lr 1e-3   --out_dir runs/oasis_unet_v1 --num_classes 4   --val_split 0.15 --test_split 0.15