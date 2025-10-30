import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int = 1337):
    import random, numpy as np, torch
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
    """
    pred = torch.softmax(pred_logits, dim=1)
    num = 2.0 * torch.sum(pred * target_onehot, dim=(0,2,3))
    den = torch.sum(pred + target_onehot, dim=(0,2,3)) + eps
    dice_per_class = (num / den).detach().cpu().numpy()  # (C,)
    return dice_per_class

def iou_from_logits(pred_logits, target, num_classes: int):
    """
    target: (N,H,W) int64 class ids
    """
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1)  # (N,H,W)
        ious = []
        for c in range(num_classes):
            p = (pred == c)
            t = (target == c)
            inter = (p & t).sum().item()
            union = (p | t).sum().item()
            if union == 0:
                ious.append(1.0)  # ignore empty class
            else:
                ious.append(inter / union)
        return np.array(ious)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_curves(log_path, out_path):
    with open(log_path, "r") as f:
        hist = json.load(f)
    epochs = list(range(1, len(hist["train_loss"]) + 1))

    # Loss plot
    plt.figure()
    plt.plot(epochs, hist["train_loss"], label="train_loss")
    plt.plot(epochs, hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.savefig(out_path.replace(".png", "_loss.png"), bbox_inches="tight")
    plt.close()

    # Dice plot (mean)
    plt.figure()
    plt.plot(epochs, hist["train_dice_mean"], label="train_dice_mean")
    plt.plot(epochs, hist["val_dice_mean"], label="val_dice_mean")
    plt.xlabel("Epoch"); plt.ylabel("Mean Dice"); plt.legend(); plt.title("Dice")
    plt.savefig(out_path.replace(".png", "_dice.png"), bbox_inches="tight")
    plt.close()