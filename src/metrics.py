"""
metrics.py
----------
Segmentation evaluation metrics.

Per-class DSC and IoU are accumulated as voxel counts across batches so the
final result is a true volume-weighted score (not an average of per-batch
scores, which would bias toward small volumes).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class SegMetricAccumulator:
    """Accumulates per-class true positives, false positives, false negatives
    across an entire validation set, then computes DSC and IoU at the end.

    Usage:
        acc = SegMetricAccumulator(n_classes=3)
        for batch in loader:
            logits = model(batch.image)
            pred = logits.argmax(dim=1)  # multi-class
            acc.update(pred, batch.mask)
        results = acc.compute()
        # {'dsc_class_1': 0.95, 'iou_class_1': 0.91, ...}
    """

    n_classes: int
    binary: bool = False  # True for Stage 1 (single-channel sigmoid output)
    tp: List[float] = field(default_factory=list)
    fp: List[float] = field(default_factory=list)
    fn: List[float] = field(default_factory=list)

    def __post_init__(self):
        # In binary mode we track a single foreground class (class index 1)
        n = 2 if self.binary else self.n_classes
        self.tp = [0.0] * n
        self.fp = [0.0] * n
        self.fn = [0.0] * n

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred: (B, D, H, W) integer predictions in [0, n_classes-1]
        target: (B, D, H, W) integer ground truth in [0, n_classes-1]
        """
        pred = pred.long()
        target = target.long()
        n = len(self.tp)
        for c in range(n):
            p_c = pred == c
            t_c = target == c
            self.tp[c] += float((p_c & t_c).sum().item())
            self.fp[c] += float((p_c & ~t_c).sum().item())
            self.fn[c] += float((~p_c & t_c).sum().item())

    def compute(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        n = len(self.tp)
        dscs, ious = [], []
        for c in range(n):
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            dsc = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            results[f"dsc_class_{c}"] = dsc
            results[f"iou_class_{c}"] = iou
            if c > 0:  # skip background for the mean
                dscs.append(dsc)
                ious.append(iou)
        results["dsc_mean_fg"] = sum(dscs) / max(len(dscs), 1)
        results["iou_mean_fg"] = sum(ious) / max(len(ious), 1)
        return results

    def reset(self) -> None:
        n = len(self.tp)
        self.tp = [0.0] * n
        self.fp = [0.0] * n
        self.fn = [0.0] * n
