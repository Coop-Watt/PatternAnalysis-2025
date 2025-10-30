import os
import argparse
import glob
import nibabel as nib
import numpy as np
import torch
from modules import UNet2D
from dataset import normalize_img

def save_mask_png(mask, out_path):
    # saves as uint8 indexed-ish; change as needed
    from PIL import Image
    Image.fromarray(mask.astype(np.uint8)).save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet2D(in_channels=1, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.nii*")))
    assert len(image_paths) > 0, "No images found"

    with torch.no_grad():
        for p in image_paths:
            ni = nib.load(p)
            img = ni.get_fdata(caching='unchanged')
            if img.ndim == 3:
                img = img[..., 0]
            img = normalize_img(img)
            x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)  # (1,1,H,W)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H,W)

            out_png = os.path.join(args.out_dir, os.path.basename(p).replace(".nii.gz", "_pred.png").replace(".nii", "_pred.png"))
            save_mask_png(pred, out_png)
            print(f"Saved {out_png}")

if __name__ == "__main__":
    main()