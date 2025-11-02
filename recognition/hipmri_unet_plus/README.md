# HipMRI U-Net+ (Cooper Richardson-Watt)

## Problem & Approach
One paragraph on HipMRI segmentation + one paragraph on your U-Net variant and loss/metrics.

## Data & Preprocessing
Paths (read-only shared), any intensity normalisation/resampling; cite sources if applicable. Justify split (e.g., 70/15/15). 

## Environment / Dependencies
Python, PyTorch, nibabel, scikit-image, tqdm, etc. List versions.

## How to Run
### Local smoke test
python train.py --images_dir /home/groups/.../semantic_MRs \
  --labels_dir /home/groups/.../semantic_labels_only \
  --epochs 1 --batch_size 2 --num_classes 6 --out_dir runs/smoke

### Full run on Rangpur (Slurm)
sbatch recognition/hipmri_unet_plus/scripts/train_hipmri.slurm

### Predict
python predict.py --images_dir ... --labels_dir ... --checkpoint PATH/TO/best.pt --out_dir runs/preds

## Results
- Best val mean Dice: …
- Test Dice per class: …
- Curves: (embed small PNG)
- Qualitative overlays: (embed small PNG)

## Notes on Reproducibility
Seeds, deterministic flags, GPU used.

