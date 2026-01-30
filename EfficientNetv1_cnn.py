import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# --------- Utility: Stochastic Depth (DropPath) ---------

class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in residual branch)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # (B, 1, 1, 1) mask, broadcast along C,H,W
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor


# --------- SE Block (Squeeze-and-Excitation) ---------

class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


# --------- Gated depthwise conv (your gated conv, depthwise) ---------

class GatedDepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv_feat = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.conv_gate = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(x))
        out = feat * gate
        out = self.bn(out)
        out = self.act(out)
        return out


# --------- MBConv-style block with Gated DW + SE + DropPath ---------

class GatedMBConvSE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        expansion: int = 4,
        kernel_size: int = 3,
        se_reduction: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.exp_channels = in_channels * expansion
        padding = kernel_size // 2

        # 1x1 expansion
        self.expand = nn.Conv2d(in_channels, self.exp_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.exp_channels)
        self.act1 = nn.SiLU(inplace=True)

        # Gated depthwise conv
        self.gated_dw = GatedDepthwiseConv2d(self.exp_channels, kernel_size=kernel_size, padding=padding)

        # SE
        self.se = SEBlock(self.exp_channels, reduction=se_reduction)

        # 1x1 projection
        self.project = nn.Conv2d(self.exp_channels, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x

        # Expansion
        out = self.expand(x)
        out = self.bn1(out)
        out = self.act1(out)

        # Gated depthwise conv (keeps H,W)
        out = self.gated_dw(out)

        # SE
        out = self.se(out)

        # Projection
        out = self.project(out)
        out = self.bn2(out)

        # Residual + stochastic depth
        out = self.drop_path(out) + identity
        return out


# --------- Top-level 4-layer inpainting net ---------

class GatedEfficientInpaint4(nn.Module):
    """
    4-block CNN for inpainting with EfficientNet-style tricks:
      - stem conv
      - 3x GatedMBConvSE blocks (MBConv + SE + SiLU + DropPath)
      - final plain conv to RGB

    All convs have stride=1 and 'same' padding -> spatial dimensions preserved.
    """
    def __init__(
        self,
        in_channels: int = 3,     # e.g. RGB (3)
        base_channels: int = 32,  # "B0 baseline" width
        width_mult: float = 1.0,  # EfficientNet-style width scaling
        drop_connect_rate: float = 0.1,  # max stochastic depth for last block
    ):
        super().__init__()

        # --- compound-style width scaling (simple version) ---
        c = int(round(base_channels * width_mult))

        # Stem (not counted among the 3 gated blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # Stochastic depth schedule across the 3 MBConv blocks (linearly increasing)
        dpr = [drop_connect_rate * i / max(1, (3 - 1)) for i in range(3)]

        # 3 GatedMBConvSE blocks with kernels [3, 5, 5]
        self.block1 = GatedMBConvSE(
            in_channels=c,
            expansion=2,       # lighter first block
            kernel_size=3,
            se_reduction=4,
            drop_path=dpr[0],
        )
        self.block2 = GatedMBConvSE(
            in_channels=c,
            expansion=4,
            kernel_size=5,
            se_reduction=4,
            drop_path=dpr[1],
        )
        self.block3 = GatedMBConvSE(
            in_channels=c,
            expansion=4,
            kernel_size=5,
            se_reduction=4,
            drop_path=dpr[2],
        )

        # Final head conv: NOT gated (your 4th layer)
        self.head = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: (B, in_channels, H, W)  e.g. concat([masked_rgb, mask], dim=1)
        returns: (B, 3, H, W)
        """
        x = self.stem(x)
        x = self.block1(x)   # gated
        x = self.block2(x)   # gated
        x = self.block3(x)   # gated
        x = self.head(x)     # plain conv
        return x

class GatedEfficientInpaint4_WithLatent(nn.Module):
    """
    Same as GatedEfficientInpaint4, but:
      - accepts AE latent embedding
      - upsamples latent to image resolution
      - concatenates with image
      - uses 1x1 conv to project back to 3 channels
    """

    def __init__(
        self,
        in_channels: int = 3,      # RGB image
        latent_channels: int = 256,
        base_channels: int = 32,
        width_mult: float = 1.0,
        drop_connect_rate: float = 0.1,
    ):
        super().__init__()

        # -------------------------------------------------
        # Latent fusion (NEW)
        # -------------------------------------------------
        self.latent_fuse = nn.Conv2d(
            in_channels + latent_channels,
            in_channels,
            kernel_size=1,
            bias=True
        )

        # --- compound-style width scaling ---
        c = int(round(base_channels * width_mult))

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # Stochastic depth schedule
        dpr = [drop_connect_rate * i / max(1, (3 - 1)) for i in range(3)]

        # Gated MBConv blocks
        self.block1 = GatedMBConvSE(
            in_channels=c,
            expansion=2,
            kernel_size=3,
            se_reduction=4,
            drop_path=dpr[0],
        )
        self.block2 = GatedMBConvSE(
            in_channels=c,
            expansion=4,
            kernel_size=5,
            se_reduction=4,
            drop_path=dpr[1],
        )
        self.block3 = GatedMBConvSE(
            in_channels=c,
            expansion=4,
            kernel_size=5,
            se_reduction=4,
            drop_path=dpr[2],
        )

        # Head
        self.head = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, latent):
        """
        x      : (B, 3, H, W)        masked image
        latent : (B, C_lat, h, w)   AE latent embedding
        """

        # -------------------------------------------------
        # Latent conditioning (NEW)
        # -------------------------------------------------
        latent = F.interpolate(
            latent,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, latent], dim=1)   # (B, 3 + C_lat, H, W)
        x = self.latent_fuse(x)              # (B, 3, H, W)

        # -------------------------------------------------
        # Original network (UNCHANGED)
        # -------------------------------------------------
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)

        return x

