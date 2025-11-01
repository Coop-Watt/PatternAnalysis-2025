import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset

_try_pil = None
_try_nib = None


def _load_png(path):
    """Load a grayscale PNG as float32 array."""
    global _try_pil
    if _try_pil is None:
        from PIL import Image
        _try_pil = Image
    img = _try_pil.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def _load_nifti(path):
    """Load a NIfTI file as float32 array; squeeze last dim if singleton."""
    global _try_nib
    if _try_nib is None:
        import nibabel as nib
        _try_nib = nib
    arr = _try_nib.load(path).get_fdata()
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _discover_pairs(images_dir, labels_dir):
    """
    Match image/label files by basename for either PNG or NIfTI.
    Returns: ([(img_path, lbl_path), ...], mode) where mode in {"png","nifti"}
    """
    # Prefer PNG if present
    pngs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if pngs:
        pairs = []
        for ip in pngs:
            lp = os.path.join(labels_dir, os.path.basename(ip))
            if not os.path.exists(lp):
                raise FileNotFoundError(f"Missing label for {ip}: expected {lp}")
            pairs.append((ip, lp))
        return pairs, "png"

    # Otherwise NIfTI
    niis = sorted(glob.glob(os.path.join(images_dir, "*.nii"))) + \
           sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if niis:
        pairs = []
        for ip in niis:
            base = os.path.basename(ip)
            lp = os.path.join(labels_dir, base)
            if not os.path.exists(lp):
                raise FileNotFoundError(f"Missing label for {ip}: expected {lp}")
            pairs.append((ip, lp))
        return pairs, "nifti"

    raise FileNotFoundError(f"No .png or .nii(.gz) files under {images_dir}")


class NiftiSeg2DDataset(Dataset):
    """
    2D slice segmentation dataset with optional PNG/NIfTI I/O, z-score norm,
    optional flips, and deterministic train/val/test splitting.

    Returns:
      img: torch.float32 tensor, shape [1, H, W]
      lbl: torch.int64 tensor,  shape [H, W]   (class ids)
    """

    def __init__(
        self,
        images_dir,
        labels_dir,
        split=None,              # "train" | "val" | "test" | None
        val_split=0.15,
        test_split=0.15,
        split_seed=1337,
        augment=False,
        num_classes=2,
        focus_label=None,
        one_hot=False,           # accepted but not used (CE loss expects class ids)
        **kwargs,
    ):
        self.pairs, self.mode = _discover_pairs(images_dir, labels_dir)
        self.augment = augment
        self.num_classes = int(num_classes)
        self.focus_label = focus_label
        self.one_hot = one_hot  # kept for API compatibility

        # Build deterministic split indices
        n = len(self.pairs)
        if (split is None) or (val_split is None) or (test_split is None):
            self.idxs = list(range(n))
        else:
            rng = np.random.RandomState(int(split_seed))
            perm = rng.permutation(n).tolist()

            n_test = int(round(n * float(test_split)))
            n_val  = int(round(n * float(val_split)))
            n_test = min(max(n_test, 0), n)
            n_val  = min(max(n_val, 0), max(n - n_test, 0))
            n_train = max(n - n_test - n_val, 0)

            test_idx  = perm[:n_test]
            val_idx   = perm[n_test:n_test + n_val]
            train_idx = perm[n_test + n_val:n_test + n_val + n_train]

            if split == "train":
                self.idxs = train_idx
            elif split == "val":
                self.idxs = val_idx
            elif split == "test":
                self.idxs = test_idx
            else:
                # Fallback: full dataset if unknown token
                self.idxs = list(range(n))

    def __len__(self):
        return len(self.idxs)

    # ------------ low-level loaders ------------
    def _load_pair(self, ipath, lpath):
        if self.mode == "png":
            img = _load_png(ipath)
            lbl = _load_png(lpath).astype(np.uint8)
            # OASIS PNG masks are 0 or 255 → map to {0,1}
            if lbl.max() > 1:
                lbl = (lbl > 0).astype(np.int64)
            else:
                lbl = lbl.astype(np.int64)
        else:
            img = _load_nifti(ipath)
            lbl = _load_nifti(lpath).astype(np.float32)
            # Assume integer classes for NIfTI labels; cast to int64
            lbl = lbl.round().astype(np.int64)
        return img, lbl

    def _maybe_augment(self, img, lbl):
        # simple flips (copy to keep arrays contiguous)
        if np.random.rand() < 0.5:
            img = np.flip(img, 1).copy()
            lbl = np.flip(lbl, 1).copy()
        if np.random.rand() < 0.5:
            img = np.flip(img, 0).copy()
            lbl = np.flip(lbl, 0).copy()
        return img, lbl

    # ------------ public API ------------
    def __getitem__(self, i):
        ipath, lpath = self.pairs[self.idxs[i]]
        img, lbl = self._load_pair(ipath, lpath)

        # Optional binary focus (e.g., isolate a single label id)
        if self.focus_label is not None:
            lbl = (lbl == int(self.focus_label)).astype(np.int64)

        # Per-slice z-score normalization
        m = float(img.mean())
        s = float(img.std()) + 1e-8
        img = (img - m) / s

        if self.augment:
            img, lbl = self._maybe_augment(img, lbl)

        # To contiguous tensors
        img = np.ascontiguousarray(img[None, ...].astype(np.float32))  # [1,H,W]
        lbl = np.ascontiguousarray(lbl.astype(np.int64))               # [H,W]
        return torch.from_numpy(img), torch.from_numpy(lbl)
