import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

_try_pil = None
_try_nib = None

# NEW: tolerant matcher — handles ".nii.png" images vs ".png"/"_seg.png"/"_mask.png" labels
def _find_label_for_image(img_basename: str, labels_dir: str) -> str:
    base = img_basename
    # If image is "name.nii.png", normalise to "name.png"
    if base.endswith(".nii.png"):
        base_core = base[:-8] + ".png"   # drop ".nii", keep ".png"
    else:
        base_core = base

    candidates = [
        os.path.join(labels_dir, base_core),                              # e.g., case_001_slice_0.png
        os.path.join(labels_dir, base),                                   # exact (in case labels also have .nii.png)
        os.path.join(labels_dir, base_core.replace(".png", "_seg.png")),  # ..._seg.png
        os.path.join(labels_dir, base_core.replace(".png", "_mask.png")), # ..._mask.png
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Missing label for {img_basename}: tried -> " + ", ".join(candidates)
    )

def _load_png(path):
    global _try_pil
    if _try_pil is None:
        from PIL import Image
        _try_pil = Image
    img = _try_pil.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def _load_nifti(path):
    global _try_nib
    if _try_nib is None:
        import nibabel as nib
        _try_nib = nib
    arr = _try_nib.load(path).get_fdata()
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)

def _discover_pairs(images_dir, labels_dir):
    # PNG path
    pngs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if pngs:
        pairs = []
        for ip in pngs:
            b = os.path.basename(ip)
            lp = _find_label_for_image(b, labels_dir)
            pairs.append((ip, lp))
        return pairs, "png"

    # NIfTI path
    niis = sorted(glob.glob(os.path.join(images_dir, "*.nii"))) + \
           sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if niis:
        lbls = [os.path.join(labels_dir, os.path.basename(p)) for p in niis]
        return list(zip(niis, lbls)), "nifti"

    raise FileNotFoundError(f"No .png or .nii(.gz) files under {images_dir}")

class NiftiSeg2DDataset(Dataset):
    def __init__(self, images_dir, labels_dir,
                 split=None, val_split=None, test_split=None,
                 augment=False, num_classes=2, focus_label=None, **kwargs):
        self.pairs, self.mode = _discover_pairs(images_dir, labels_dir)
        self.augment = augment
        self.num_classes = num_classes
        self.focus_label = focus_label

    def __len__(self):
        return len(self.pairs)

    def _load(self, ipath, lpath):
        if self.mode == "png":
            img = _load_png(ipath)
            lbl = _load_png(lpath).astype(np.uint8)
            if lbl.max() > 1:
                lbl = (lbl > 0).astype(np.int64)
        else:
            img = _load_nifti(ipath)
            lbl = _load_nifti(lpath).astype(np.int64)
        return img, lbl

    def __getitem__(self, idx):
        ipath, lpath = self.pairs[idx]
        img, lbl = self._load(ipath, lpath)

        if self.focus_label is not None:
            lbl = (lbl == int(self.focus_label)).astype(np.int64)

        m, s = float(img.mean()), float(img.std()) + 1e-8
        img = (img - m) / s

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, 1).copy()
                lbl = np.flip(lbl, 1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, 0).copy()
                lbl = np.flip(lbl, 0).copy()

        img = np.ascontiguousarray(img[None, ...].astype(np.float32))  # [1,H,W]
        lbl = np.ascontiguousarray(lbl.astype(np.int64))               # [H,W]
        return torch.from_numpy(img), torch.from_numpy(lbl)
