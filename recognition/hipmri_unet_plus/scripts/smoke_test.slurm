#!/bin/bash
#SBATCH --job-name=oasis_smoke
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH -o slurm_%j.out

set -euxo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"

python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir -r requirements.txt

python - << 'PY'
import torch
print("compiled CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
PY

python train.py \
  --images_dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --labels_dir /home/groups/comp3710/OASIS/keras_png_slices_seg_train \
  --epochs 1 --batch_size 2 --lr 1e-3 \
  --out_dir runs/oasis_png_smoke --num_classes 2 \
  --val_split 0.1 --test_split 0.1 --num_workers 2
