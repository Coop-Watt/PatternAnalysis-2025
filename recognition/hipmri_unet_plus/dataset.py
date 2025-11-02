import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["NiftiSeg2DDataset", "_discover_pairs"]

_try_pil = None
_try_nib = None

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

# --- HipMRI mapping: *_LFOV.nii.gz (image) -> *_SEMANTIC.nii.gz (label) ---
def _find_label_for_image(img_basename: str, labels_dir: str) -> str:
    import os
    cands = []
    if img_basename.endswith("_LFOV.nii.gz"):
        cands.append(os.path.join(
            labels_dir, img_basename.replace("_LFOV.nii.gz", "_SEMANTIC.nii.gz")
        ))
    # Fallback: identical basename
    cands.append(os.path.join(labels_dir, img_basename))
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Missing label for {img_basename}: tried -> " + ", ".join(cands))

# HipMRI-aware discover_pairs that always uses the mapping helper above
def _discover_pairs(images_dir: str, labels_dir: str):
    import os, glob
    nii_imgs = sorted(glob.glob(os.path.join(images_dir, "*.nii"))) + \
               sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    png_imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if nii_imgs:
        mode = "nifti"
        imgs = nii_imgs
    elif png_imgs:
        mode = "png"
        imgs = png_imgs
    else:
        raise FileNotFoundError(f"No images found in {images_dir}")
    pairs = []
    for ip in imgs:
        b = os.path.basename(ip)
        lp = _find_label_for_image(b, labels_dir)
        pairs.append((ip, lp))
    return pairs, mode

def _normalize_img(x: np.ndarray, eps: float = 1e-6):
    x = x.astype(np.float32)
    m = float(x.mean())
    s = float(x.std())
    if s < eps:
        s = 1.0
    return (x - m) / s

def _to_channels(arr: np.ndarray, num_classes: int, dtype=np.uint8):
    h, w = arr.shape
    out = np.zeros((num_classes, h, w), dtype=dtype)
    for c in range(num_classes):
        out[c] = (arr == c).astype(dtype)
    return out

class NiftiSeg2DDataset(Dataset):
    """
    Returns:
      image: FloatTensor (1, H, W) normalized per-slice
      mask : LongTensor (H, W) class ids (or one-hot if one_hot=True)
    """
    def __init__(
        self,
        images_dir,
        labels_dir,
        split=None, val_split=None, test_split=None,
        augment=False, num_classes=2, focus_label=None, one_hot=False, **kwargs
    ):
        self.pairs, self.mode = _discover_pairs(images_dir, labels_dir)
        self.augment = augment
        self.num_classes = num_classes
        self.focus_label = focus_label
        self.one_hot = one_hot

        # NOTE: split/val_split/test_split are accepted for API compatibility
        # with train.py but are not used here (we load all pairs).

    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, ipath, lpath):
        if self.mode == "png":
            img = _load_png(ipath)
            lbl = _load_png(lpath).astype(np.uint8)
            # common OASIS mask: 0/255 -> 0/1
            if lbl.max() > 1:
                lbl = (lbl > 0).astype(np.int64)
            else:
                lbl = lbl.astype(np.int64)
        else:
            img = _load_nifti(ipath)
            lbl = _load_nifti(lpath).astype(np.int64)
        return img, lbl

    def __getitem__(self, idx):
        ipath, lpath = self.pairs[idx]
        img, lbl = self._load_pair(ipath, lpath)

        if self.focus_label is not None:
            lbl = (lbl == int(self.focus_label)).astype(np.int64)

        img = _normalize_img(img)

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, 1).copy()
                lbl = np.flip(lbl, 1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, 0).copy()
                lbl = np.flip(lbl, 0).copy()

        img = np.ascontiguousarray(img[None, ...].astype(np.float32))  # (1,H,W)
        img_t = torch.from_numpy(img)

        if self.one_hot:
            mask = _to_channels(lbl, self.num_classes, dtype=np.uint8)
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float()
        else:
            mask_t = torch.from_numpy(np.ascontiguousarray(lbl)).long()

        return img_t, mask_t# --- HipMRI mapping override: *_LFOV.nii.gz (image) -> *_SEMANTIC.nii.gz (label) ---
def _find_label_for_image(img_basename: str, labels_dir: str) -> str:
    import os
    candidates = []
    if img_basename.endwith("_LFOV.nii.gz"):
        candidates.append(os.path.join(labels_dir,
                                       img_basename.replace("LFOV.nii.gz", "_SEMANTIC.nii.gz")))
    # Fallback: identical basename (only if labels really share the same name)
    candidates.append(os.path.join(labels_dir, img_basename))

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Missing label for {img_basename}: tried -> " + ",".join(candidates)
    )
# --- HipMRI mapping override: *_LFOV.nii.gz (image) -> *_SEMANTIC.nii.gz (label) ---
def _find_label_for_image(img_basename: str, labels_dir: str) -> str:
    import os
    candidates = []
    if img_basename.endswith("_LFOV.nii.gz"):
        candidates.append(os.path.join(labels_dir,
                                       img_basename.replace("_LFOV.nii.gz", "_SEMANTIC.nii.gz")))
    # Fallback: identical basename (only if labels really share the same name)
    candidates.append(os.path.join(labels_dir, img_basename))

    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Missing label for {img_basename}: tried -> " + ",".join(candidates)
    )
