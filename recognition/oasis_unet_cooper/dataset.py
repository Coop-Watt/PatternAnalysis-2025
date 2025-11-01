# --- helpers at top of dataset.py ---

import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset

def _strip_ext(basename: str) -> str:
    # remove trailing .nii.png or .png
    if basename.endswith(".nii.png"):
        return basename[:-8]
    if basename.endswith(".png"):
        return basename[:-4]
    return basename

def _alt_stems(stem: str):
    """
    Return possible alternative stems for a label given an image stem.
    Handles case_###_slice_k  -> seg_###_slice_k
            (and .nii-in-stem edge case)
    """
    stems = {stem}
    # if someone included ".nii" in stem (rare), include version without it
    if stem.endswith(".nii"):
        stems.add(stem[:-4])

    # case_ -> seg_
    def _segify(s):
        if s.startswith("case_"):
            return "seg_" + s[len("case_"):]
        return None

    for s in list(stems):
        alt = _segify(s)
        if alt:
            stems.add(alt)
        if s.endswith(".nii"):
            alt2 = _segify(s[:-4])
            if alt2:
                stems.add(alt2)

    return list(stems)

_SUFFIXES = ("", "_seg", "_mask")

def _find_label_for_image(img_basename: str, labels_dir: str):
    stem = _strip_ext(img_basename)
    candidates = []
    for st in _alt_stems(stem):
        # try plain + common suffixes, with both .png and .nii.png
        for suf in _SUFFIXES:
            candidates.append(os.path.join(labels_dir, st + suf + ".png"))
            candidates.append(os.path.join(labels_dir, st + suf + ".nii.png"))

    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p); uniq.append(p)

    for p in uniq:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Missing label for {img_basename}: tried -> " + ", ".join(os.path.basename(x) for x in uniq[:8])
    )

def _discover_pairs(images_dir, labels_dir):
    pngs = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if pngs:
        pairs = []
        for ip in pngs:
            b = os.path.basename(ip)
            lp = _find_label_for_image(b, labels_dir)
            pairs.append((ip, lp))
        return pairs, "png"

    niis = sorted(glob.glob(os.path.join(images_dir, "*.nii"))) + \
           sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if niis:
        pairs = []
        for ip in niis:
            b = os.path.basename(ip)
            # if you ever export labels as NIfTI, add a nifti-aware finder here
            lp = _find_label_for_image(b.replace(".nii.gz", ".nii.png").replace(".nii", ".nii.png"), labels_dir)
            pairs.append((ip, lp))
        return pairs, "nifti"

    raise FileNotFoundError(f"No .png or .nii(.gz) files under {images_dir}")
