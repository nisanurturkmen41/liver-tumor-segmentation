"""
dataset.py
----------
Patch-based 3D dataset for the cascaded liver-tumor pipeline.

Why patches?
    Processed isotropic volumes are large (~360x360x375), and a full volume
    won't fit in 16 GB VRAM at training resolution. We sample 96x96x96 voxel
    patches on the fly.

Why foreground-biased sampling?
    Random patch sampling would yield mostly empty/background patches because
    liver+tumor occupy roughly 3% of the volume, tumor alone ~0.01%. We
    bias 2/3 of patches to be centered on a foreground voxel so the network
    sees the target structure regularly.

Stage 1 (this file's default):
    Labels are binarised: {1 liver, 2 tumor} -> 1 (foreground),
    {0 background} -> 0. Stage 1 only needs to localize the liver envelope.

Stage 2 (use stage='stage2'):
    Labels are kept as {0, 1, 2}. Patches are sampled inside the liver
    bounding box only (see stage2_bbox in __getitem__).

Augmentations are kept lightweight to avoid CPU bottleneck on Kaggle (only
2 workers reliable). Heavy elastic deformation is off by default; intensity
shift + flip + rotation are cheap and effective.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI as float32 numpy array. Assumes Week 1 preprocessing already done."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def compute_liver_bbox(mask: np.ndarray, margin: int = 8) -> Optional[Tuple[slice, slice, slice]]:
    """Bounding box of liver+tumor voxels with a small margin. Returns None if empty."""
    fg = mask > 0
    if not fg.any():
        return None
    coords = np.argwhere(fg)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, mask.shape)
    return tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))


def random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
    foreground_bias: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop a random patch. With probability `foreground_bias`, the patch is
    centered on a random foreground voxel."""
    D, H, W = image.shape
    pd, ph, pw = patch_size

    # Pad if the volume is smaller than the patch along any axis (rare after
    # resampling, but defensive).
    pad = [
        (0, max(0, pd - D)),
        (0, max(0, ph - H)),
        (0, max(0, pw - W)),
    ]
    if any(p[1] > 0 for p in pad):
        image = np.pad(image, pad, mode="constant", constant_values=0.0)
        mask = np.pad(mask, pad, mode="constant", constant_values=0)
        D, H, W = image.shape

    use_fg = (rng.random() < foreground_bias) and (mask > 0).any()
    if use_fg:
        # Sample center from a uniformly chosen foreground voxel
        fg_coords = np.argwhere(mask > 0)
        cz, cy, cx = fg_coords[rng.integers(0, len(fg_coords))]
    else:
        cz = rng.integers(pd // 2, D - pd // 2 + 1)
        cy = rng.integers(ph // 2, H - ph // 2 + 1)
        cx = rng.integers(pw // 2, W - pw // 2 + 1)

    # Clamp so the patch stays inside the volume
    z0 = int(np.clip(cz - pd // 2, 0, D - pd))
    y0 = int(np.clip(cy - ph // 2, 0, H - ph))
    x0 = int(np.clip(cx - pw // 2, 0, W - pw))

    img_patch = image[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    msk_patch = mask[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
    return img_patch, msk_patch


def augment_patch(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight on-the-fly augmentation. CPU-cheap."""
    # Random flips along each spatial axis
    for axis in range(3):
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis)
            mask = np.flip(mask, axis=axis)

    # 90-degree rotations in axial plane (preserves voxel grid alignment)
    k = int(rng.integers(0, 4))
    if k:
        image = np.rot90(image, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(1, 2))

    # Random intensity shift in [-0.05, 0.05] (image is normalized to [0,1])
    shift = rng.uniform(-0.05, 0.05)
    image = image + shift

    # Ensure contiguous after flips/rotations (PyTorch needs contiguous arrays)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class LiverPatchDataset(Dataset):
    """
    Patch-based dataset for Stage 1 (liver localization) or Stage 2 (tumor
    refinement inside liver bbox).

    Args:
        processed_dir: directory containing liver_<id>_processed.nii.gz
                       and liver_<id>_label_processed.nii.gz
        patient_ids: list of patient ID strings (e.g. ["0", "1", ...])
        patch_size: (D, H, W) patch size
        samples_per_volume: how many random patches to draw per volume per
                            epoch (since one volume yields many patches)
        stage: 'stage1' (binary liver) or 'stage2' (3-class tumor inside liver)
        foreground_bias: probability of sampling a foreground-centered patch
        augment: enable training-time augmentation
        cache_volumes: if True, keep loaded volumes in RAM (faster but uses
                       more memory). Default False because 40 GB > Kaggle RAM.
    """

    def __init__(
        self,
        processed_dir: str,
        patient_ids: List[str],
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        samples_per_volume: int = 4,
        stage: str = "stage1",
        foreground_bias: float = 0.66,
        augment: bool = True,
        cache_volumes: bool = False,
        seed: Optional[int] = None,
    ):
        assert stage in ("stage1", "stage2")
        self.processed_dir = Path(processed_dir)
        self.patient_ids = list(patient_ids)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.stage = stage
        self.foreground_bias = foreground_bias
        self.augment = augment
        self.cache_volumes = cache_volumes
        self._cache: dict = {}
        self._seed = seed

    def __len__(self) -> int:
        return len(self.patient_ids) * self.samples_per_volume

    def _paths(self, pid: str) -> Tuple[Path, Path]:
        img = self.processed_dir / f"liver_{pid}_processed.nii.gz"
        lbl = self.processed_dir / f"liver_{pid}_label_processed.nii.gz"
        return img, lbl

    def _load(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        if self.cache_volumes and pid in self._cache:
            return self._cache[pid]
        img_path, lbl_path = self._paths(pid)
        image = load_nifti(img_path)
        mask = load_nifti(lbl_path).astype(np.uint8)
        if self.cache_volumes:
            self._cache[pid] = (image, mask)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pid = self.patient_ids[idx // self.samples_per_volume]
        # Worker-aware RNG so each DataLoader worker draws different patches
        # and seeds are reproducible across epochs if needed.
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed_components = [idx, worker_id]
        if self._seed is not None:
            seed_components.append(self._seed)
        rng = np.random.default_rng(
            int(np.uint32(hash(tuple(seed_components)) & 0xFFFFFFFF))
        )

        image, mask = self._load(pid)

        if self.stage == "stage2":
            # Restrict sampling to the liver bounding box
            bbox = compute_liver_bbox(mask, margin=8)
            if bbox is not None:
                image = image[bbox]
                mask = mask[bbox]
            # Use the multi-class mask as-is (0/1/2)
            target_mask = mask
        else:
            # Stage 1: binary foreground vs background
            target_mask = (mask > 0).astype(np.uint8)

        img_patch, msk_patch = random_crop(
            image, target_mask, self.patch_size, self.foreground_bias, rng
        )
        if self.augment:
            img_patch, msk_patch = augment_patch(img_patch, msk_patch, rng)

        # Add channel dimension. Image: (1, D, H, W) float32. Mask: (D, H, W) int64.
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        msk_t = torch.from_numpy(msk_patch).long()
        return img_t, msk_t


class FullVolumeDataset(Dataset):
    """For validation/inference: returns the full volume (image, mask, pid).
    No patching, no augmentation. Use with batch_size=1."""

    def __init__(self, processed_dir: str, patient_ids: List[str], stage: str = "stage1"):
        assert stage in ("stage1", "stage2")
        self.processed_dir = Path(processed_dir)
        self.patient_ids = list(patient_ids)
        self.stage = stage

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        img_path = self.processed_dir / f"liver_{pid}_processed.nii.gz"
        lbl_path = self.processed_dir / f"liver_{pid}_label_processed.nii.gz"
        image = load_nifti(img_path)
        mask = load_nifti(lbl_path).astype(np.uint8)
        if self.stage == "stage1":
            mask = (mask > 0).astype(np.uint8)
        img_t = torch.from_numpy(image).unsqueeze(0).float()
        msk_t = torch.from_numpy(mask).long()
        return img_t, msk_t, pid
