import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Sparse weight mixin (for prune-and-grow) ----------

class SparseWeightMixin:
    """
    Mixin for conv layers: supports a binary mask over weights.
    - During forward: uses weight * mask.
    - mask is a buffer so you can modify it with a prune-and-grow scheduler.
    """
    def init_mask(self):
        w = self.weight.data
        mask = torch.ones_like(w)
        self.register_buffer("weight_mask", mask)

    @property
    def effective_weight(self):
        if hasattr(self, "weight_mask"):
            return self.weight * self.weight_mask
        return self.weight


# ---------- Decomposed SLaK-style large kernel conv ----------

class DecomposedLargeKernelDWConv(nn.Module):
    """
    SLaK-style decomposition of a large MxM kernel into two
    sparse/grouped convolutions of shape MxN and NxM. :contentReference[oaicite:3]{index=3}

    We use:
      - group convs (sparse groups, more width),
      - optional binary masks for prune-and-grow.

    All convs are depthwise-style: groups = channels // group_width.
    """
    def __init__(
        self,
        channels: int,
        k_long: int = 21,
        k_short: int = 5,
        group_width: int = 1,     # e.g. 1 → depthwise, 4 → grouped
        use_sparse_mask: bool = True,
    ):
        super().__init__()
        groups = max(1, channels // group_width)

        class _SparseConv2d(nn.Conv2d, SparseWeightMixin):
            def __init__(self, *args, **kwargs):
                nn.Conv2d.__init__(self, *args, **kwargs)
                SparseWeightMixin.__init__(self)
                self.init_mask()

            def forward(self, x):
                w = self.effective_weight
                return F.conv2d(
                    x, w, self.bias, self.stride,
                    self.padding, self.dilation, self.groups
                )

        ConvClass = _SparseConv2d if use_sparse_mask else nn.Conv2d

        # M x N conv (tall)
        self.conv_tall = ConvClass(
            channels, channels,
            kernel_size=(k_long, k_short),
            padding=(k_long // 2, k_short // 2),
            groups=groups,
            bias=False,
        )
        self.bn_tall = nn.BatchNorm2d(channels)

        # N x M conv (wide)
        self.conv_wide = ConvClass(
            channels, channels,
            kernel_size=(k_short, k_long),
            padding=(k_short // 2, k_long // 2),
            groups=groups,
            bias=False,
        )
        self.bn_wide = nn.BatchNorm2d(channels)

    def forward(self, x):
        out_tall = self.bn_tall(self.conv_tall(x))
        out_wide = self.bn_wide(self.conv_wide(x))
        return out_tall + out_wide


# ---------- Gated SLaK block (used for first 3 blocks) ----------

class GatedSLaKBlock(nn.Module):
    """
    SLaK-style large kernel block with:
      - decomposed large-grouped conv (M×N + N×M),
      - gate branch (3x3 depthwise),
      - 1x1 conv + BN + GELU,
      - residual shortcut.
    """
    def __init__(
        self,
        channels: int,
        k_long: int = 21,
        k_short: int = 5,
        group_width: int = 1,
        use_sparse_mask: bool = True,
    ):
        super().__init__()
        self.channels = channels

        # Decomposed large kernel conv
        self.lk_conv = DecomposedLargeKernelDWConv(
            channels,
            k_long=k_long,
            k_short=k_short,
            group_width=group_width,
            use_sparse_mask=use_sparse_mask,
        )

        # Gate: 3x3 depthwise
        self.gate_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # Pointwise conv for channel mixing
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        feat = self.lk_conv(x)
        gate = torch.sigmoid(self.gate_conv(x))
        out = feat * gate

        out = self.pw(out)
        out = self.bn(out)
        out = self.act(out)

        return out + identity   # identity shortcut


# ---------- 4-layer SLaK-style inpainting CNN ----------

class GatedSLaKInpaint4(nn.Module):
    """
    4-block SLaK-inspired CNN for image inpainting.

    Layout:
      - Stem:   3x3 conv (3 -> C), stride 1, keeps H,W. (NOT gated)
      - Block1: GatedSLaKBlock
      - Block2: GatedSLaKBlock
      - Block3: GatedSLaKBlock
      - Block4: Plain 3x3 conv (C -> 3), stride 1. (NOT gated)

    All layers preserve spatial dimensions (no pooling, no strided convs).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        width_mult: float = 1.3,   # “expand width” factor; tune as needed
        k_long: int = 21,
        k_short: int = 5,
        group_width: int = 1,
        use_sparse_mask: bool = True,
    ):
        super().__init__()
        C = int(round(base_channels * width_mult))

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # 3 gated SLaK blocks
        self.block1 = GatedSLaKBlock(
            C, k_long=k_long, k_short=k_short,
            group_width=group_width, use_sparse_mask=use_sparse_mask
        )
        self.block2 = GatedSLaKBlock(
            C, k_long=k_long, k_short=k_short,
            group_width=group_width, use_sparse_mask=use_sparse_mask
        )
        self.block3 = GatedSLaKBlock(
            C, k_long=k_long, k_short=k_short,
            group_width=group_width, use_sparse_mask=use_sparse_mask
        )

        # Head (NOT gated)
        self.head = nn.Conv2d(
            C, 3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        """
        x: (B, 3, H, W) masked/occluded RGB
        -> (B, 3, H, W) inpainted RGB
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x
