"""
losses.py
---------
Loss functions for the cascaded liver-tumor pipeline.

Why hybrid Dice + CE?
    Pure Dice loss focuses on overlap but has unstable gradients when the
    foreground is tiny (tumor ~0.01% of voxels). Pure CE is overwhelmed by
    background. The sum gives stable gradients from CE plus the overlap
    optimization from Dice.

For Stage 1 (binary):
    Use DiceBCELoss with `binary=True`. Network outputs 1 channel of logits.

For Stage 2 (multi-class):
    Use DiceCELoss with `n_classes=3`. Network outputs 3 channels of logits.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert integer label tensor (B, D, H, W) to one-hot (B, C, D, H, W)."""
    return F.one_hot(target, num_classes=n_classes).permute(0, 4, 1, 2, 3).float()


class DiceBCELoss(nn.Module):
    """For binary segmentation. Logits shape: (B, 1, D, H, W). Target: (B, D, H, W) in {0,1}."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_f = target.float().unsqueeze(1)  # (B, 1, D, H, W)
        bce = self.bce(logits, target_f)

        probs = torch.sigmoid(logits)
        dims = (0, 2, 3, 4)
        inter = (probs * target_f).sum(dims)
        denom = probs.sum(dims) + target_f.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


class DiceCELoss(nn.Module):
    """For multi-class. Logits: (B, C, D, H, W). Target: (B, D, H, W) integer in [0,C-1].

    Args:
        n_classes: number of classes
        ce_weight: weighting between CE and Dice
        class_weights: optional per-class weights for CE (helps with imbalance)
        include_background_in_dice: if False, the Dice term ignores class 0.
            Recommended True for Stage 2 to push tumor (class 2) hard.
    """

    def __init__(
        self,
        n_classes: int = 3,
        ce_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
        include_background_in_dice: bool = False,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.ce_weight = ce_weight
        self.include_background_in_dice = include_background_in_dice
        self.smooth = smooth
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_classes),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights)

        probs = F.softmax(logits, dim=1)
        target_oh = _one_hot(target, self.n_classes)

        if not self.include_background_in_dice:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        dims = (0, 2, 3, 4)
        inter = (probs * target_oh).sum(dims)
        denom = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + (1.0 - self.ce_weight) * dice_loss
