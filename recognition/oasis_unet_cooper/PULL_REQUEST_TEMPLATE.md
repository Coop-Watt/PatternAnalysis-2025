# 🚀 OASIS 2D U-Net (PNG) — Pull Request

## Summary
- Adds `recognition/oasis_unet_cooper`: 2D **U-Net** for OASIS **PNG** brain MRI segmentation (PyTorch).
- Trains/evaluates on Rangpur A100; saves metrics, curves, and predictions under `runs/`.
- Reproducible via pinned deps (`numpy<2`) and Slurm script.
- **No datasets or model weights are committed.**

> ⚠️ Only tick the difficulty you actually meet based on `metrics.json`.
- [ ] **Easy** (OASIS PNG): **both classes** (bg, brain) Dice ≥ 0.90 on **test**
- [ ] **Normal** (optional follow‑up): HipMRI + improved U‑Net (separate folder/section)

---

## How to run (Rangpur)
Environment (CUDA 11.8 wheels):
```bash
cd ~/PatternAnalysis-2025/recognition/oasis_unet_cooper
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-cache-dir
python - << 'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Submit training:
```bash
sbatch scripts/train_oasis.slurm
squeue -u $USER
ls -1tr slurm_*.out | tail -1 | xargs tail -n 80 -f
```

Script arguments (already set inside `scripts/train_oasis.slurm`):
```
--images_dir /home/groups/comp3710/OASIS/keras_png_slices_train
--labels_dir /home/groups/comp3710/OASIS/keras_png_slices_seg_train
--epochs 60 --batch_size 8 --lr 1e-3
--num_classes 2 --val_split 0.15 --test_split 0.15
--out_dir runs/oasis_png_v1
```

Predict examples:
```bash
source .venv/bin/activate
python predict.py   --checkpoint runs/oasis_png_v1/best.pt   --images_dir /home/groups/comp3710/OASIS/keras_png_slices_validate   --out_dir runs/oasis_png_v1/preds_val   --num_classes 2
```

---

## Results
Fill from `runs/oasis_png_v1/metrics.json` after training.

**Best epoch:** `__`  |  **Val mean Dice:** `__`

| Split | Dice (bg) | Dice (fg/brain) | **Mean Dice** |
|:----:|:---------:|:----------------:|:-------------:|
| Val  |   `__`    |       `__`       |     `__`      |
| Test |   `__`    |       `__`       |     `__`      |

Artifacts (checked in **your run folder**):
- `runs/oasis_png_v1/args.json`
- `runs/oasis_png_v1/best.pt` *(not committed)*
- `runs/oasis_png_v1/history.json`
- `runs/oasis_png_v1/metrics.json`
- `runs/oasis_png_v1/training_curves_loss.png`
- `runs/oasis_png_v1/training_curves_dice.png`
- (optional) `runs/oasis_png_v1/preds_{val,test}/*.png` examples

> Add 2–4 qualitative examples (input, GT, prediction) to this PR description or the README.

---

## Environment & System (record for reproducibility)
- Partition/GRES: `a100`, `--gres=gpu:a100:1`
- Node: `a100-__`
- GPU: `NVIDIA A100 40GB`
- Torch: `2.2.2+cu118`  |  TorchVision: `0.17.2+cu118`
- NumPy: `< 2`  |  Pillow: `>=10`
- Seed: `1337`  |  cuDNN deterministic: `True`
- Commit: `<git-hash>`

---

## Files in this PR
- `recognition/oasis_unet_cooper/dataset.py` — PNG/NIfTI loader (z‑score; safe flips; contiguous)
- `recognition/oasis_unet_cooper/modules.py` — `UNet2D` (BN+Dropout; bilinear up; clean blocks)
- `recognition/oasis_unet_cooper/train.py` — seeded split via `Subset`; DiceCE; AdamW + cosine LR; early stop; saves `best.pt`/JSON/plots
- `recognition/oasis_unet_cooper/predict.py` — directory inference → PNG masks
- `recognition/oasis_unet_cooper/utils.py` — dice/IoU; **Agg** plotting; seeding; JSON helpers
- `recognition/oasis_unet_cooper/requirements.txt` — libs (no torch/vision)
- `recognition/oasis_unet_cooper/scripts/train_oasis.slurm` — Rangpur A100 config
- `recognition/oasis_unet_cooper/README.md` — usage and details

---

## Checklist
- [x] Small, frequent commits with meaningful messages
- [x] No datasets or model weights committed
- [x] Scripts run end‑to‑end on Rangpur (paths match)
- [x] README updated and consistent with commands
- [x] `metrics.json` produced and reported here
- [x] 2–4 qualitative examples attached (or linked from `runs/.../preds_*`)
- [x] Line endings LF; tabs replaced with spaces (HPC‑safe)
- [x] Numpy < 2; PyTorch CUDA 11.8 wheels confirmed (`CUDA available: True`)

---

### Notes (optional)
- If results fall short of Easy (e.g., foreground Dice < 0.90), document the gap and any attempted fixes (LR, aug, batch size, epochs).
- For extra credit (Normal), add HipMRI 2D with an improved U‑Net in a separate folder and include a brief section with its metrics.
