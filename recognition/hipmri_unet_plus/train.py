import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from modules import UNet2D
from dataset import NiftiSeg2DDataset
from utils import set_seed, soft_dice_score, iou_from_logits, save_json, plot_curves

class DiceCELoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce(logits, target)  # target: (N,H,W)
        with torch.no_grad():
            target_oh = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        num = 2.0 * torch.sum(probs * target_oh, dim=(0, 2, 3))
        den = torch.sum(probs + target_oh, dim=(0, 2, 3)) + self.eps
        dice_per_class = num / den
        dice_loss = 1.0 - dice_per_class.mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss

def evaluate(model, loader, device, num_classes):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    dice_accum, iou_accum = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            loss = ce(logits, masks)

            target_oh = torch.nn.functional.one_hot(masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
            dice_pc = soft_dice_score(logits, target_oh)                     # (C,)
            iou_pc  = iou_from_logits(logits, masks, num_classes=num_classes) # (C,)

            total_loss += loss.item() * imgs.size(0)
            dice_accum.append(dice_pc)
            iou_accum.append(iou_pc)

    n = len(loader.dataset)
    loss_avg = total_loss / max(n, 1)
    dice_mean = np.mean(np.vstack(dice_accum), axis=0) if dice_accum else np.zeros(num_classes)
    iou_mean  = np.mean(np.vstack(iou_accum),  axis=0) if iou_accum  else np.zeros(num_classes)
    return loss_avg, dice_mean, iou_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", default="runs/oasis_png_v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_classes", type=int, default=2)       # OASIS PNG = 2 classes
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--test_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--early_stop_patience", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Deterministic split into train/val/test using indices ----
    base_eval = NiftiSeg2DDataset(args.images_dir, args.labels_dir,
                                  augment=False, num_classes=args.num_classes)
    base_train = NiftiSeg2DDataset(args.images_dir, args.labels_dir,
                                   augment=True, num_classes=args.num_classes)

    n = len(base_eval)
    n_test = int(round(n * args.test_split))
    n_val  = int(round(n * args.val_split))
    n_train = max(n - n_test - n_val, 0)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    val_idx  = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_ds = Subset(base_train, train_idx)
    val_ds   = Subset(base_eval,  val_idx)
    test_ds  = Subset(base_eval,  test_idx)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ---- Model, optim, sched, loss ----
    model = UNet2D(in_channels=1, num_classes=args.num_classes,
                   base_ch=args.base_ch, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceCELoss(num_classes=args.num_classes, dice_weight=0.5, ce_weight=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice_mean": [], "val_dice_mean": [],
        "val_dice_per_class": [],
    }

    best_val = -1.0
    patience_left = args.early_stop_patience
    best_ckpt = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_dice = []

        pbar = tqdm(train_ld, total=len(train_ld), desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # monitor mean dice over classes
            probs = torch.softmax(logits, dim=1)
            target_oh = torch.nn.functional.one_hot(masks, num_classes=args.num_classes).permute(0, 3, 1, 2).float()
            dice_pc = (2.0 * torch.sum(probs * target_oh, dim=(0, 2, 3)) /
                       (torch.sum(probs + target_oh, dim=(0, 2, 3)) + 1e-6))
            running_dice.append(dice_pc.detach().cpu().numpy())
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / max(len(train_ds), 1)
        train_dice_mean = float(np.mean(np.vstack(running_dice), axis=0).mean()) if running_dice else 0.0
        history["train_loss"].append(train_loss)
        history["train_dice_mean"].append(train_dice_mean)

        # Validation
        val_loss, val_dice_pc, _ = evaluate(model, val_ld, device, args.num_classes)
        val_dice_mean = float(val_dice_pc.mean())
        history["val_loss"].append(val_loss)
        history["val_dice_mean"].append(val_dice_mean)
        history["val_dice_per_class"].append(val_dice_pc.tolist())

        # Save history
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        scheduler.step()

        # Early stopping + checkpointing
        improved = val_dice_mean > best_val
        if improved:
            best_val = val_dice_mean
            patience_left = args.early_stop_patience
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_left -= 1

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_dice={train_dice_mean:.4f} val_dice={val_dice_mean:.4f} "
              f"(best={best_val:.4f}, patience_left={patience_left})")

        if patience_left <= 0:
            print("Early stopping triggered.")
            break

    # Final test evaluation on best model
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_dice_pc, test_iou_pc = evaluate(model, test_ld, device, args.num_classes)
    results = {
        "best_val_mean_dice": best_val,
        "test_loss": float(test_loss),
        "test_dice_per_class": test_dice_pc.tolist(),
        "test_dice_mean": float(test_dice_pc.mean()),
        "test_iou_per_class": test_iou_pc.tolist(),
        "test_iou_mean": float(test_iou_pc.mean())
    }
    save_json(results, os.path.join(args.out_dir, "metrics.json"))
    print("Test Results:", json.dumps(results, indent=2))

    # Curves
    try:
        plot_curves(os.path.join(args.out_dir, "history.json"),
                    os.path.join(args.out_dir, "training_curves.png"))
    except Exception as e:
        print("Curve plotting failed:", e)

if __name__ == "__main__":
    main()
