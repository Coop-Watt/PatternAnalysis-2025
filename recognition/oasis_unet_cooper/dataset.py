import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

def normalize_img(x: np.ndarray, eps=1e-6):
    x = x.astype(np.float32)
    mean = x.mean()
    std = x.std()
    if std < eps:
        std = 1.0
    return (x - mean) / std

def to_channels(arr: np.ndarray, num_classes: int, dtype=np.uint8):
    # assumes labels are 0..C-1
    h, w = arr.shape
    out = np.zeros((num_classes, h, w), dtype=dtype)
    for c in range(num_classes):
        out[c] = (arr == c).astype(dtype)
    return out

class NiftiSeg2DDataset(Dataset):
    """
    Expects parallel directory structure:
      images_dir/*.nii.gz
      labels_dir/*.nii.gz
    Matching is by sorted filename order. Adjust as needed.

    Returns tensors:
      image: (1, H, W)
      mask : (H, W) long (class ids) OR one-hot (C, H, W) if one_hot=True
    """
    def __init__(self, images_dir, labels_dir, split="train",
                 split_seed=1337, val_split=0.15, test_split=0.15,
                 one_hot=False, num_classes=4, augment=True):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.nii*")))
        self.labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii*")))
        assert len(self.images) == len(self.labels) and len(self.images) > 0, "No pairs found"
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.augment = augment if split == "train" else False

        # split
        idxs = list(range(len(self.images)))
        random.Random(split_seed).shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_split)
        n_val  = int(n * val_split)
        test_idx = idxs[:n_test]
        val_idx  = idxs[n_test:n_test+n_val]
        train_idx= idxs[n_test+n_val:]

        if split == "train":
            self.idxs = train_idx
        elif split == "val":
            self.idxs = val_idx
        else:
            self.idxs = test_idx

    def __len__(self):
        return len(self.idxs)

    def _maybe_augment(self, img, mask):
        # simple flips
        import numpy as np
        if random.random() < 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        return img, mask

    def __getitem__(self, i):
        real_i = self.idxs[i]
        img_nii = nib.load(self.images[real_i])
        lbl_nii = nib.load(self.labels[real_i])

        img = img_nii.get_fdata(caching='unchanged')
        lbl = lbl_nii.get_fdata(caching='unchanged')

        # handle [H,W] or [H,W,1]
        if img.ndim == 3:
            img = img[..., 0]
        if lbl.ndim == 3:
            lbl = lbl[..., 0]

        img = normalize_img(img)
        lbl = lbl.astype(np.int64)

        if self.augment:
            img, lbl = self._maybe_augment(img, lbl)

        # to tensors
        img_t = torch.from_numpy(img).unsqueeze(0).float()  # (1,H,W)
        if self.one_hot:
            mask = to_channels(lbl, self.num_classes, dtype=np.uint8)  # (C,H,W)
            mask_t = torch.from_numpy(mask).float()
        else:
            mask_t = torch.from_numpy(lbl).long()  # (H,W)
        return img_t, mask_t