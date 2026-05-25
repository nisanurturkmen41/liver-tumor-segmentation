"""
train_stage1.py
---------------
Train the Stage 1 (liver localization) 3D U-Net.

Designed for Kaggle:
  - Saves checkpoint every epoch -> resume after timeout
  - Mixed precision (AMP) -> fits comfortably on P100/T4 16GB
  - Per-epoch validation with sliding-window inference + DSC/IoU logging
  - All logs saved as JSON for later plotting

Usage (local CLI):
    python -m src.train_stage1 \
        --processed_dir /path/to/processed \
        --folds_json folds.json \
        --fold 0 \
        --out_dir runs/stage1_fold0 \
        --epochs 80 \
        --batch_size 2

Inside a Kaggle notebook, import main() and call it with an args namespace.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LiverPatchDataset, FullVolumeDataset
from src.losses import DiceBCELoss
from src.metrics import SegMetricAccumulator
from src.models.unet3d import UNet3D, count_parameters
from src.inference import sliding_window_inference


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--folds_json", type=str, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--samples_per_volume", type=int, default=4)
    ap.add_argument("--base_features", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=5,
                    help="Run full-volume validation every N epochs (expensive)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_dsc: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_dsc": best_dsc,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt.get("best_dsc", 0.0)


@torch.no_grad()
def validate(model, val_loader, device, patch_size) -> Dict[str, float]:
    """Full-volume sliding-window validation. Returns DSC/IoU on Stage 1 binary.

    FullVolumeDataset returns image (1, D, H, W) and mask (D, H, W). The
    DataLoader (batch_size=1) adds a batch dim, giving:
        image: (1, 1, D, H, W)
        mask:  (1, D, H, W)
    No extra unsqueeze needed.
    """
    model.eval()
    acc = SegMetricAccumulator(n_classes=2, binary=True)
    pbar = tqdm(val_loader, desc="val", leave=False)
    for img, mask, _pid in pbar:
        img = img.to(device)
        mask = mask.to(device)
        logits = sliding_window_inference(
            img, model, patch_size=tuple(patch_size), n_classes=1, device=device
        )
        pred = (torch.sigmoid(logits[:, 0]) > 0.5).long()  # (1, D, H, W)
        acc.update(pred, mask)
    return acc.compute()



def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    with open(args.folds_json) as f:
        folds = json.load(f)
    fold_key = f"fold_{args.fold}"
    train_ids = folds[fold_key]["train"]
    val_ids = folds[fold_key]["val"]
    print(f"Fold {args.fold}: {len(train_ids)} train, {len(val_ids)} val")

    train_set = LiverPatchDataset(
        processed_dir=args.processed_dir,
        patient_ids=train_ids,
        patch_size=tuple(args.patch_size),
        samples_per_volume=args.samples_per_volume,
        stage="stage1",
        foreground_bias=0.66,
        augment=True,
        seed=args.seed,
    )
    val_set = FullVolumeDataset(
        processed_dir=args.processed_dir, patient_ids=val_ids, stage="stage1"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )

    # --- Model / loss / optim ---
    model = UNet3D(
        in_channels=1, out_channels=1, base_features=args.base_features, levels=4
    ).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    loss_fn = DiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch = 0
    best_dsc = 0.0
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        start_epoch, best_dsc = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, device
        )
        start_epoch += 1

    log_path = out_dir / "log.json"
    history = []
    if log_path.exists():
        history = json.loads(log_path.read_text())

    # --- Training loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for img, mask in pbar:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(img)
                loss = loss_fn(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        mean_loss = float(np.mean(losses))
        elapsed = time.time() - epoch_start
        log_row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }

        # --- Validation ---
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_loader, device, args.patch_size)
            log_row.update({f"val_{k}": v for k, v in val_metrics.items()})
            val_dsc = val_metrics.get("dsc_class_1", 0.0)
            if val_dsc > best_dsc:
                best_dsc = val_dsc
                save_checkpoint(
                    out_dir / "best.pt", model, optimizer, scaler, epoch, best_dsc
                )
                print(f"  ** new best DSC (liver) = {best_dsc:.4f} **")

        history.append(log_row)
        log_path.write_text(json.dumps(history, indent=2))

        # Always save last checkpoint for Kaggle timeout safety
        save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, epoch, best_dsc)

        print(
            f"epoch {epoch:3d} | loss {mean_loss:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
            + (f" | val DSC {log_row.get('val_dsc_class_1', float('nan')):.4f}"
               if 'val_dsc_class_1' in log_row else "")
        )

    print(f"Training done. Best val DSC (liver) = {best_dsc:.4f}")


if __name__ == "__main__":
    main()
