## Summary
- Adds `recognition/oasis_unet_cooper`: 2D U-Net for OASIS brain segmentation (PyTorch).
- Meets Easy difficulty target: all labels Dice >= 0.90 on test split (see `metrics.json`).
- Includes training script, evaluation, and README with reproducibility details.

## How to run
See `recognition/oasis_unet_cooper/README.md` for environment, training, and inference commands.
Do **not** commit datasets or model weights.

## Results
- Metrics and curves are saved under `runs/oasis_unet_v1/`.
- Include mean/per-class Dice and example masks in the README section of your PR.

## Checklist
- [x] Small, frequent commits with meaningful messages.
- [x] No datasets or model weights committed.
- [x] Scripts run end-to-end on Rangpur paths.
- [x] README is properly formatted.