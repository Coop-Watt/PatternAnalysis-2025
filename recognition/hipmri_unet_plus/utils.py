import os
import json
import torch
import numpy as np

# Use non-interactive backend for headless servers (Rangpur)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_dice_score(pred_logits, target_onehot, eps=1e-6):
    """
    pred_logits: (N,C,H,W) raw scores
    target_onehot: (N,C,H,W) float in {0,1}
    returns: np.ndarray of shape (C,)
    """
    pred = torch.softmax(pred_logits, dim=1)
    num = 2.0 * torch.sum(pred * target_onehot, dim=(0, 2, 3))
    den = torch.sum(pred + target_onehot, dim=(0, 2, 3)) + eps
    dice_per_class = (num / den).detach().cpu().numpy()  # (C,)
    return dice_per_class


def iou_from_logits(logits: torch.Tensor,
                    target: torch.Tensor,
                    num_classes: int,
                    eps: float = 1e-6) -> np.ndarray:
    """
    Per-class IoU from logits and integer masks.
    Returns np.array shape (C,) with NaN for classes absent in both pred & gt.
    """
    with torch.no_grad():
        pred = logits.argmax(dim=1)  # (N,H,W)
        ious = []
        for c in range(num_classes):
            pred_c = (pred == c)
            targ_c = (target == c)
            inter = (pred_c & targ_c).sum().item()
            union = (pred_c | targ_c).sum().item()
            if union == 0:
                ious.append(np.nan)          # ignore absent class
            else:
                ious.append(inter / (union + eps))
        return np.asarray(ious, dtype=np.float32)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_curves(log_path, out_path):
    with open(log_path, "r") as f:
        hist = json.load(f)

    train_loss = hist.get("train_loss", [])
    val_loss = hist.get("val_loss", [])
    train_dice = hist.get("train_dice_mean", [])
    val_dice = hist.get("val_dice_mean", [])
    epochs = list(range(1, max(len(train_loss), len(val_loss), len(train_dice), len(val_dice)) + 1))

    # Loss plot
    if train_loss or val_loss:
        plt.figure()
        if train_loss: plt.plot(range(1, len(train_loss)+1), train_loss, label="train_loss")
        if val_loss:   plt.plot(range(1, len(val_loss)+1),   val_loss,   label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
        plt.savefig(out_path.replace(".png", "_loss.png"), bbox_inches="tight")
        plt.close()

    # Dice plot (mean)
    if train_dice or val_dice:
        plt.figure()
        if train_dice: plt.plot(range(1, len(train_dice)+1), train_dice, label="train_dice_mean")
        if val_dice:   plt.plot(range(1, len(val_dice)+1),   val_dice,   label="val_dice_mean")
        plt.xlabel("Epoch"); plt.ylabel("Mean Dice"); plt.legend(); plt.title("Dice")
        plt.savefig(out_path.replace(".png", "_dice.png"), bbox_inches="tight")
        plt.close()
