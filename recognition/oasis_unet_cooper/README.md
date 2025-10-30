# OASIS 2D Brain Segmentation with U-Net (PyTorch)

**Goal:** Solve the Easy-difficulty task: *Segment the 2D OASIS brain dataset* with a 2D U‑Net (or Improved U‑Net) such that **all labels have Dice ≥ 0.90** on the held-out test split.

This repository folder is structured for direct inclusion under `recognition/` in the PatternAnalysis-2025 repo.

---

## How it works (overview)
- **Model:** A clean 2D U‑Net with batch norm + dropout; configurable channels and depth.
- **Data:** NIfTI images and labels (2D slices). Normalization per-slice; optional on‑the‑fly class-channel conversion.
- **Training:** Cross‑entropy + soft Dice loss; cosine LR annealing; early stopping by validation Dice; mixed precision for speed (if CUDA).
- **Evaluation:** Per-class and mean Dice/IoU; training curves saved to `runs/`.

---

## Quickstart (Rangpur HPC / local)

> **Important:** Do **not** commit datasets or model `.pt` weights to your fork.

```bash
# (1) Create env
module load python/3.10  # if on HPC; adjust per site
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# (2) Train (adjust paths)
python train.py   --images_dir /home/groups/comp3710/OASIS/images   --labels_dir /home/groups/comp3710/OASIS/labels   --epochs 60 --batch_size 8 --lr 1e-3   --out_dir runs/oasis_unet_v1 --num_classes 4   --val_split 0.15 --test_split 0.15

# (3) Predict on a directory of images
python predict.py   --checkpoint runs/oasis_unet_v1/best_model.pt   --images_dir /home/groups/comp3710/OASIS/images   --out_dir runs/oasis_unet_v1/preds
```

If you use SLURM, submit `scripts/train_oasis.slurm`. If your site uses PBS, adapt the script header accordingly.

---

## Repo structure (required by task)
- `modules.py` – model components (2D U‑Net + building blocks)
- `dataset.py` – NIfTI 2D slice dataset + transforms
- `train.py` – training/validation/testing loop, metrics & plots
- `predict.py` – inference on a folder of images; saves predicted masks
- `utils.py` – dice/iou metrics, plotting, seed utilities
- `requirements.txt` – pinned major deps
- `scripts/train_oasis.slurm` – example SLURM batch (adapt for PBS if needed)
- `README.md` – this file

---

## Reproducibility
- Deterministic seeds set where feasible (`--seed`).
- Check and record: PyTorch, CUDA, cuDNN versions; GPU model/VRAM in your README results section.
- Save `args.json` and `metrics.json` in the run folder.

---

## Results to include in the report
- Mean Dice and per‑class Dice on test set (target: **≥ 0.90** for all classes).
- Loss and Dice curves (`runs/*/training_curves.png`).
- A couple of visual examples of predicted masks vs ground truth.

---

## Notes
- If time permits, consider extending to **HipMRI 2D** with Improved U-Net/CAN to reach Normal difficulty (update `--num_classes`, paths, and label handling).
- Keep commits small with meaningful messages (design, dataset, trainer, first pass results, tuning, final polish).