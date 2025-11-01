#!/bin/bash
#SBATCH --job-name=oasis_unet_png
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH -o slurm_%j.out

set -euxo pipefail

# Always run from the submission directory
cd "$SLURM_SUBMIT_DIR"

# Python env
module load python/3.10 || true
python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install CUDA 11.8 PyTorch *before* the rest to avoid CPU wheels
pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Other deps (numpy<2 etc.)
pip install --no-cache-dir -r requirements.txt

# Headless plots for HPC
export MPLBACKEND=Agg

# Quick sanity check
python - << 'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

# Train on PNG slices
python train.py \
  --images_dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --labels_dir /home/groups/comp3710/OASIS/keras_png_slices_seg_train \
  --epochs 60 --batch_size 8 --lr 1e-3 \
  --out_dir runs/oasis_png_v1 --num_classes 2 \
  --val_split 0.15 --test_split 0.15
