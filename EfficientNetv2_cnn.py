import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------- Stochastic Depth (DropPath) ----------

class DropPath(nn.Module):
    """Stochastic depth per sample (residual branch)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


# ---------- SE Block (lighter, higher reduction ratio) ----------

class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)          # ReLU is fine here
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


# ---------- Gated Fused-MBConv (Blocks 1 & 2) ----------

class GatedFusedMBConv(nn.Module):
    """
    Fused-MBConv-style block:
      - gated 3x3 conv (in -> expand_channels)
      - SE
      - 1x1 projection (expand -> out)
      - ReLU in fused stage
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 2,
        kernel_size: int = 3,
        se_reduction: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_channels = in_channels * expansion
        padding = kernel_size // 2

        # Gated 3x3 conv (fused: expansion + spatial)
        self.conv_feat = nn.Conv2d(
            in_channels,
            self.expand_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.conv_gate = nn.Conv2d(
            in_channels,
            self.expand_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(self.expand_channels)
        self.act1 = nn.ReLU(inplace=True)   # ReLU in fused stage

        # SE on expanded channels (lighter)
        self.se = SEBlock(self.expand_channels, reduction=se_reduction)

        # Projection
        self.project = nn.Conv2d(
            self.expand_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Use residual only if shape matches
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x

        # Gated fused conv
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        out = feat * gate
        out = self.bn1(out)
        out = self.act1(out)

        # SE
        out = self.se(out)

        # Projection
        out = self.project(out)
        out = self.bn2(out)

        if self.use_residual:
            out = self.drop_path(out) + identity

        return out


# ---------- Gated MBConv (Block 3) ----------

class GatedMBConv(nn.Module):
    """
    MBConv-style block:
      - 1x1 expansion (SiLU)
      - gated depthwise 3x3
      - SE
      - 1x1 projection
      - residual + DropPath
    """
    def __init__(
        self,
        in_channels: int,
        expansion: int = 3,
        kernel_size: int = 3,
        se_reduction: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.expand_channels = in_channels * expansion
        padding = kernel_size // 2

        # 1x1 expansion
        self.expand = nn.Conv2d(
            in_channels,
            self.expand_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.expand_channels)
        self.act1 = nn.SiLU(inplace=True)   # SiLU in MBConv

        # Gated depthwise conv
        self.dw_feat = nn.Conv2d(
            self.expand_channels,
            self.expand_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=self.expand_channels,
            bias=False,
        )
        self.dw_gate = nn.Conv2d(
            self.expand_channels,
            self.expand_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=self.expand_channels,
            bias=True,
        )
        self.bn2 = nn.BatchNorm2d(self.expand_channels)
        self.act2 = nn.SiLU(inplace=True)

        # SE
        self.se = SEBlock(self.expand_channels, reduction=se_reduction)

        # 1x1 projection
        self.project = nn.Conv2d(
            self.expand_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.expand(x)
        out = self.bn1(out)
        out = self.act1(out)

        feat = self.dw_feat(out)
        gate = torch.sigmoid(self.dw_gate(out))
        out = feat * gate
        out = self.bn2(out)
        out = self.act2(out)

        out = self.se(out)

        out = self.project(out)
        out = self.bn3(out)

        out = self.drop_path(out) + identity
        return out


# ---------- Top-level 4-block EfficientNetV2-style inpainting CNN ----------

class GatedEfficientV2Inpaint4(nn.Module):
    """
    4-block CNN for image inpainting with EfficientNetV2 ideas:
      - Block 1: Gated Fused-MBConv (ReLU)
      - Block 2: Gated Fused-MBConv (ReLU)
      - Block 3: Gated MBConv (SiLU, depthwise + SE)
      - Block 4: Plain 3x3 conv to RGB

    All convs use stride=1 + same padding -> spatial dimensions preserved.
    """
    def __init__(
        self,
        in_channels: int = 3,        # masked RGB only
        base_channels: int = 32,     # "B0-like" base width
        width_mult: float = 1.0,     # EfficientNetV2-style width scaling
        drop_connect_rate: float = 0.1,
        se_reduction: int = 8,
    ):
        super().__init__()

        # Simple compound scaling: scale all channels with width_mult
        c = int(round(base_channels * width_mult))

        # Stochastic depth schedule across the 3 gated blocks
        dpr = [drop_connect_rate * i / max(1, (3 - 1)) for i in range(3)]
        # e.g. [0.0, 0.05, 0.1] when drop_connect_rate=0.1

        # Block 1: Fused-MBConv-like, 3x3, expansion=2, ReLU
        self.block1 = GatedFusedMBConv(
            in_channels=in_channels,
            out_channels=c,
            expansion=2,
            kernel_size=3,
            se_reduction=se_reduction,
            drop_path=dpr[0],
        )

        # Block 2: Fused-MBConv-like, 5x5, expansion=2, ReLU
        self.block2 = GatedFusedMBConv(
            in_channels=c,
            out_channels=c,
            expansion=2,
            kernel_size=5,
            se_reduction=se_reduction,
            drop_path=dpr[1],
        )

        # Block 3: MBConv-like, 3x3 depthwise, expansion=3, SiLU
        self.block3 = GatedMBConv(
            in_channels=c,
            expansion=3,
            kernel_size=3,
            se_reduction=se_reduction,
            drop_path=dpr[2],
        )

        # Block 4: Plain head conv (NOT gated)
        self.head = nn.Conv2d(
            c,
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) masked/occluded RGB
        returns: (B, 3, H, W) inpainted RGB
        """
        x = self.block1(x)  # gated fused
        x = self.block2(x)  # gated fused
        x = self.block3(x)  # gated MBConv
        x = self.head(x)    # plain conv
        return x
