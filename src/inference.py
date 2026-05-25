"""
inference.py
------------
Sliding-window inference over a full volume.

During training we feed patches; at validation/test time we need predictions
for the entire volume. We slide a (D, H, W) patch across the volume with
overlap, run the network on each patch, and aggregate by averaging the
overlapping logits (weighted by a Gaussian importance map that downweights
edge predictions where context is missing).
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_importance_map(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Build a 3D Gaussian centered on the patch. Used to downweight edges."""
    coords = [torch.linspace(-1, 1, s) for s in patch_size]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=0)
    sigma = sigma_scale * 2  # half the range, scaled
    gauss = torch.exp(-(grid ** 2).sum(dim=0) / (2 * sigma ** 2))
    gauss = gauss / gauss.max()
    return gauss.clamp(min=1e-3)  # avoid zero weights


@torch.no_grad()
def sliding_window_inference(
    image: torch.Tensor,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    n_classes: int = 1,
    device: str = "cuda",
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Args:
        image: (1, 1, D, H, W) float tensor (single volume, batch dim)
        model: trained network
        patch_size: must match training patch size for best results
        overlap: fraction of patch overlap (0.5 -> step = patch/2)
        n_classes: output channels of the model (1 for Stage 1, 3 for Stage 2)
    Returns:
        logits: (1, n_classes, D, H, W) float tensor
    """
    assert image.ndim == 5 and image.shape[0] == 1
    model.eval()
    _, _, D, H, W = image.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1 - overlap)))
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))

    # Pad volume so that patch_size fits cleanly
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
    _, _, Dp, Hp, Wp = image_padded.shape

    def stops(total: int, p: int, step: int):
        if total <= p:
            return [0]
        s = list(range(0, total - p + 1, step))
        if s[-1] != total - p:
            s.append(total - p)
        return s

    zs = stops(Dp, pd, sd)
    ys = stops(Hp, ph, sh)
    xs = stops(Wp, pw, sw)

    logit_sum = torch.zeros(
        (1, n_classes, Dp, Hp, Wp), device=device, dtype=torch.float32
    )
    weight_sum = torch.zeros((1, 1, Dp, Hp, Wp), device=device, dtype=torch.float32)

    gauss = _gaussian_importance_map(patch_size).to(device).unsqueeze(0).unsqueeze(0)

    image_padded = image_padded.to(device)
    cuda_amp = use_amp and device.startswith("cuda")

    for z in zs:
        for y in ys:
            for x in xs:
                patch = image_padded[:, :, z:z + pd, y:y + ph, x:x + pw]
                if cuda_amp:
                    with torch.cuda.amp.autocast():
                        out = model(patch)
                else:
                    out = model(patch)
                out = out.float()
                logit_sum[:, :, z:z + pd, y:y + ph, x:x + pw] += out * gauss
                weight_sum[:, :, z:z + pd, y:y + ph, x:x + pw] += gauss

    logit_avg = logit_sum / weight_sum.clamp(min=1e-8)
    # Crop back to original shape
    logit_avg = logit_avg[:, :, :D, :H, :W]
    return logit_avg
