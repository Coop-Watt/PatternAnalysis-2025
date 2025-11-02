# OASIS 2D Brain Segmentation with U-Net (PyTorch)

**Goal (Easy difficulty):** Segment the OASIS dataset with a 2D U-Net so that **all labels achieve Dice ≥ 0.90 on the held-out test split.

This folder is designed to live under:  
`recognition` in the `PatternAnalysis-2025` repo

---

## What’s inside

- **Model:** Lightweight 2D U-Net with BatchNorm + Dropout; width configurable via `--base_ch`
- **Data:** Works with **either** 2D PNG slices *(Keras-style OASIS export)* or NIfTI slices.
    - PNG pairing is auto-detected: `case_XXX_slice_Y...png` <-> `seg_XXX_Y...png`. 
- **Training:** Composite **Dice+CE** loss, AdamW, cosine LR, AMP (if CUDA), early stopping on **val Dice**.
- **Evaluation:** Per-class & mean **Dice/IoU**; curves and JSON metrics saved under `runs/`.

---

## Quickstart (Rangpur HPC)

> **Do not commit** datasets or model weights. (`.pt/.pth`) to the repo

### 0) Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r recognition/oasis_unet_cooper/requirements.txt
```

### 1) Train (PNG slices; 2 classes)

Submit the provided script (**Rangpur** (COMP3710 shared dataset):

```bash
cd recognition/oasis_unet_cooper

python train.py \
  --images_dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --labels_dir /home/groups/comp3710/OASIS/keras_png_slices_seg_train \
  --epochs 60 --batch_size 8 --lr 1e-3 \
  --out_dir runs/oasis_png_v1 \
  --num_classes 2 \
  --val_split 0.15 --test_split 0.15 \
  --num_workers 4
```

### 2) Inference (PNG or NIfTI)

predict.py loads NIfTI images then writes to PNG masks:

```bash
python predict.py \
  --checkpoint runs/oasis_png_v1/best.pt \
  --images_dir /path/to/nifti_images_for_inference \
  --out_dir runs/oasis_png_v1/preds \
  --num_classes 2
```

- PNG interface can be added easily, but the included script is NIfTI-oriented

---

## SLURM (Rangpur)

A ready-to-use job script is included: scripts/train_oasis.slurm

# submit
```bash
sbatch recognition/oasis_unet_cooper/scripts/train_oasis.slurm
```

# quick, smoke test (1 epoch on PNG) is also provided
```bash
sbatch recognition/oasis_unet_cooper/scripts/smoke_test.slurm
```

## Repo layout

|- `modules.py` — U-Net building blocks + `UNet2D`
|- `dataset.py` — Auto-detect **PNG/NIfTI**; z-score; safe flips; contiguous tensors
|- `train.py` — train/val/test loop + metrics/curve
|- `predict.py` — Inference on a folder of PNG/NIfTI images -> PNG masks
|- `utils.py` — metrics, plots, seeding helpers
|- `requirements.txt` — **No torch/vision here** (pinned deps, e.g. NumPy < 2 ect.)
|- `README.md` — this file
|- `scripts/`
    |- `train_oasis.slurm` — SLURM JOB (comp3710 partition)
    |- `smoke_test.slurm`  — short sanity test

---

## Results (PNG, 2 classes)
```bash
python train.py \
  --images_dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --labels_dir /home/groups/comp3710/OASIS/keras_png_slices_seg_train \
  --epochs 1 \
  --batch_size 2 \
  --out_dir runs/oasis_png_smoke \
  --num_classes 2 \
  --val_split 0.10 \
  --test_split 0.10
```

# Observed metrics (example runs)
            Run                   | Test Dice (per-class)  |Test Dice (mean) | Test IoU (mean)
            runs/oasis_png_smoke/ |    [0.9935, 0.9831]    |     0.9883      |     0.9818
            runs/oasis_png_v1/    |   [0.99933, 0.99826]   |     0.99880     |     0.99820


## Reproducibility & Logging

- Fixed seed via `--seed` (applies to Python/NumPy/PyTorch).
- Record environment in the PR: Python, CUDA, cuDNN, GPU model/VRAM.
- All CLI args are captured to args.json; metrics to metrics.json.

Example environment (Rangpur A100):
```makefile
Python 3.11
torch 2.2.2+cu118, torchvision 0.17.2+cu118
GPU: NVIDIA A100 40GB (CUDA 11.8)
```

Test Log:
```json
Test Results: {
  "best_val_mean_dice": 0.9987865686416626,
  "test_loss": 0.0017705535487239732,
  "test_dice_per_class": [
  0.9993329048156738,
  0.998260498046875
  ],
  "test_dice_mean": 0.9987967014312744,
  "test_iou_per_class": [
  0.9989995740205672,
  0.9973933781195532
  ],
  "test_iou_mean": 0.9981964760700601
}
```

---

## Results to report

- **Per-class Dice** and **mean Dice** on the **test** split (target for Easy: **both classes ≥ 0.90**).
- **Loss & Dice curves** (saved under `runs/...`).

> Only claim “all labels ≥ 0.90” after verifying `test_dice_per_class` in `metrics.json`.

**Run: `runs/oasis_png_smoke/ (PNG, 2 classes)**

    Split |    Dice (bg)   |    Dice (fg/brain)    |    Mean Dice
    Val   |      `__`      |          `__`         |    `0.98822`
    Test  |    `0.99347`   |       `0.98306`       |    `0.98827`

    Best epoch: `1` with **Val Dice (mean)**: `0.98822`

**Run: `runs/oasis_png_v1/ (PNG, 2 classes)**

    | Split |    Dice (bg)   |    Dice (fg/brain)    |    Mean Dice   |
    | Val   |      `__`      |          `__`         |    `0.99879`   |
    | Test  |    `0.99933`   |       `0.99826`       |    `0.99880`   |

    Best epoch: `54` with **Val Dice (mean)**: `0.99879`

---

## Troubleshooting

- **`FileNotFoundError: Missing label for ...` (PNG pairing)**
  - Ensure the two dirs are correct and case-sensitive:
    - Images: `/home/groups/comp3710/OASIS/keras_png_slices_train`
    - Labels: `/home/groups/comp3710/OASIS/keras_png_slices_seg_train`
  - Expected name pattern: `case_001_slice_0.nii.png` ↔ `seg_001_slice_0.nii.png`.
  - Quick check:
    ```bash
    ls -1 /home/groups/comp3710/OASIS/keras_png_slices_seg_train | head
    ```
    Or from Python (inside this folder):
    ```python
    from dataset import _discover_pairs
    p, m = _discover_pairs(
        "/home/groups/comp3710/OASIS/keras_png_slices_train",
        "/home/groups/comp3710/OASIS/keras_png_slices_seg_train"
    )
    print(m, len(p), p[0])
    ```

- **`CUDA available: False` on Rangpur**
  - Run on a GPU partition via SLURM (`comp3710`, `a100`, or `a100-test`).
  - Install the CUDA wheel before the rest:
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 --extra-index-url https://download.pytorch.org/whl/cu118
    ```
  - The provided SLURM scripts already print compiled CUDA, device count, and GPU name.

- **SLURM “Memory specification cannot be satisfied”**
  - Prefer `#SBATCH --mem-per-cpu=3G` with `#SBATCH --cpus-per-task=2..4`, or remove custom mem flags and use defaults for student queues.

- **`requirements.txt: No such file or directory`**
  - `cd recognition/oasis_unet_cooper` first, or pass the full path:
    ```bash
    pip install -r recognition/oasis_unet_cooper/requirements.txt
    ```

- **`ImportError: cannot import name 'NiftiSeg2DDataset'`**
  - You likely have an outdated `dataset.py`. `git pull` and ensure the class and PNG pairing helpers are present.

---
