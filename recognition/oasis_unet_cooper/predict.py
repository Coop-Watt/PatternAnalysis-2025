import os, glob, argparse
import numpy as np
import torch
from PIL import Image
from modules import UNet2D

# --- utils ---

def zscore(x, eps=1e-8):
    x = x.astype(np.float32)
    m, s = float(x.mean()), float(x.std())
    if s < eps: s = 1.0
    return (x - m) / s

def save_mask_png(mask, out_path):
    Image.fromarray(mask.astype(np.uint8)).save(out_path)

def _load_nifti(path):
    import nibabel as nib
    arr = nib.load(path).get_fdata()
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)

def _load_png(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32)

def _discover_images(images_dir):
    pngs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if pngs:
        return pngs, "png"
    niis = sorted(glob.glob(os.path.join(images_dir, "*.nii"))) + \
           sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if niis:
        return niis, "nifti"
    raise FileNotFoundError(f"No .png or .nii(.gz) images under {images_dir}")

def _safe_load_state(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            # Lightning-style ckpt: keys may be "model.*" or have prefixes
            state = {k.split("model.",1)[-1]: v for k, v in ckpt["state_dict"].items()}
            model.load_state_dict(state, strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            # Assume it is already a plain state_dict
            model.load_state_dict(ckpt, strict=False)
    else:
        # Rare case: full model was saved
        model = ckpt
    return model

# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--bilinear", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet2D(in_channels=1, num_classes=args.num_classes,
                   base_ch=args.base_ch, bilinear=args.bilinear, dropout=0.0)
    model = _safe_load_state(model, args.checkpoint, device)
    model.to(device).eval()

    files, mode = _discover_images(args.images_dir)
    print(f"Found {len(files)} {mode.upper()} images")

    with torch.no_grad():
        for p in files:
            if mode == "png":
                img = _load_png(p)
                out_name = os.path.basename(p).rsplit(".", 1)[0] + "_pred.png"
            else:
                img = _load_nifti(p)
                base = os.path.basename(p)
                if base.endswith(".nii.gz"):
                    base = base[:-7]
                elif base.endswith(".nii"):
                    base = base[:-4]
                out_name = base + "_pred.png"

            img = zscore(img)
            x = torch.from_numpy(img)[None, None, ...].float().to(device)  # [1,1,H,W]
            logits = model(x)                                              # [1,C,H,W]
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            out_path = os.path.join(args.out_dir, out_name)
            save_mask_png(pred, out_path)
            print("Saved", out_path)

if __name__ == "__main__":
    main()
