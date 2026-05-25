"""
unet3d.py
---------
Custom 3D U-Net built from scratch (no MONAI/nnU-Net dependency for the
architecture), matching the proposal specification:
  - 4 encoder/decoder levels
  - InstanceNorm3d for small-batch stability (critical: BatchNorm fails with
    batch sizes 1-2 typical for 3D medical patches)
  - LeakyReLU(alpha=0.01) to prevent the "dying ReLU" problem in deep
    volumetric networks where many activations end up negative
  - Trilinear upsampling + 1x1 conv (cheaper than transposed conv, no
    checkerboard artifacts)
  - Configurable base feature count and number of classes (1 for Stage 1
    binary, 3 for Stage 2 multi-class)

For 96^3 patches with base=32:
    Parameters: ~5.6M
    VRAM at batch=2, FP16: ~6 GB (fits easily on P100/T4 16 GB)
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive Conv3d -> InstanceNorm -> LeakyReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int, leaky_slope: float = 0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool3d(2) followed by a ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Trilinear upsample + skip concat + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # 1x1 conv to match channels after upsample, cheaper than transposed conv
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.reduce(x)
        # If shapes mismatch by 1 voxel due to odd spatial sizes, pad-align
        if x.shape[2:] != skip.shape[2:]:
            diffs = [s - x.shape[2 + i] for i, s in enumerate(skip.shape[2:])]
            # pad order is (last dim left, right, ..., first dim left, right)
            pad = []
            for d in reversed(diffs):
                pad.extend([d // 2, d - d // 2])
            x = F.pad(x, pad)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for medical volumetric segmentation.

    Args:
        in_channels: number of input channels (1 for CT)
        out_channels: number of output classes
                      Stage 1: 1 (binary, use BCE-with-logits / Dice)
                      Stage 2: 3 (background, liver, tumor; use CE/Dice)
        base_features: number of features in the first encoder block. The
                       feature count doubles every level. Use 32 for P100/T4
                       at 96^3 patches; reduce to 24 if you OOM.
        levels: number of down/up sampling stages (default 4 -> 5 levels of
                features including the bottleneck)
        leaky_slope: alpha for LeakyReLU
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        levels: int = 4,
        leaky_slope: float = 0.01,
    ):
        super().__init__()
        self.levels = levels
        feats: List[int] = [base_features * (2 ** i) for i in range(levels + 1)]
        # e.g. levels=4 -> [32, 64, 128, 256, 512]

        # Encoder
        self.input_conv = ConvBlock(in_channels, feats[0], leaky_slope=leaky_slope)
        self.downs = nn.ModuleList(
            [DownBlock(feats[i], feats[i + 1]) for i in range(levels)]
        )

        # Decoder
        self.ups = nn.ModuleList(
            [UpBlock(feats[i + 1], feats[i], feats[i]) for i in reversed(range(levels))]
        )

        # Final 1x1x1 logits head
        self.out_conv = nn.Conv3d(feats[0], out_channels, kernel_size=1)

        # Kaiming init for LeakyReLU
        self._init_weights(leaky_slope)

    def _init_weights(self, leaky_slope: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, a=leaky_slope, nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_conv(x)
        skips.append(x)
        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        x = self.downs[-1](x)  # bottleneck

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.out_conv(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = UNet3D(in_channels=1, out_channels=1, base_features=32, levels=4)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(1, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    assert y.shape == (1, 1, 96, 96, 96)
    print("UNet3D smoke test passed.")
