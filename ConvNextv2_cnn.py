import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- LayerNorm in NCHW (ConvNeXt-style) -----------------

class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors, using channels-last style."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        x_perm = x.permute(0, 2, 3, 1)       # (B, H, W, C)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)    # (B, C, H, W)


# ----------------- Global Response Normalization (GRN) -----------------
# As in ConvNeXt V2: L2 over spatial, divisive normalization over channels,
# with learnable gamma/beta and residual. :contentReference[oaicite:7]{index=7}

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)     # (B, C, 1, 1)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)  # (B, C, 1, 1)
        return self.gamma * (x * nx) + self.beta + x         # residual inside


# ----------------- Stochastic Depth (DropPath) -----------------

class DropPath(nn.Module):
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


# ----------------- Gated ConvNeXtV2-style Block (Blocks 1–3) -----------------

class GatedConvNeXtV2Block(nn.Module):
    """
    ConvNeXt V2-style block with:
      - depthwise gated 7x7 conv
      - LayerNorm2d
      - 1x1 MLP (C -> 4C -> C) with GELU + GRN
      - residual + optional DropPath
      - optional binary mask for FCMAE-style sparse conv emulation
    """
    def __init__(self, dim, mlp_ratio=4.0, kernel_size=7, drop_path=0.0):
        super().__init__()
        self.dim = dim
        padding = kernel_size // 2

        # Depthwise gated conv
        self.dw_conv_feat = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim, bias=True
        )
        self.dw_conv_gate = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding,
            groups=dim, bias=True
        )

        self.ln = LayerNorm2d(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.pw_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pw_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None):
        """
        x:    (B, C, H, W)
        mask: (B, 1, H, W) with 1=visible, 0=masked, or None
              Used only if you want FCMAE-style sparse behavior.
        """
        if mask is not None:
            x = x * mask

        residual = x

        # Depthwise gated conv
        feat = self.dw_conv_feat(x)
        gate = torch.sigmoid(self.dw_conv_gate(x))
        x = feat * gate

        # Norm + MLP + GRN
        x = self.ln(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_conv2(x)

        if mask is not None:
            x = x * mask  # dense-masked conv ~ sparse conv (submanifold)

        x = residual + self.drop_path(x)
        return x


# ----------------- 4-layer ConvNeXtV2-style Inpainting CNN -----------------

class GatedConvNeXtV2Inpaint4(nn.Module):
    """
    4-block ConvNeXtV2-inspired CNN for image inpainting.

    Blocks:
      - Stem: 3x3 conv (NOT gated), keeps H,W.
      - Block1: Gated ConvNeXtV2 block (gated depthwise, GRN).
      - Block2: Gated ConvNeXtV2 block.
      - Block3: Gated ConvNeXtV2 block.
      - Block4: Plain 3x3 conv to RGB (NOT gated).

    Features:
      - All strides=1, paddings set -> spatial dimensions preserved.
      - Optional `mask` argument lets you emulate sparse convs (FCMAE pretraining).
      - First 3 blocks are gated, 4th is standard conv.
    """
    def __init__(
        self,
        in_channels: int = 3,      # masked RGB
        base_dim: int = 32,        # width; can scale with width_mult
        width_mult: float = 1.0,
        mlp_ratio: float = 4.0,
        drop_connect_rate: float = 0.1,
        kernel_size: int = 7,
    ):
        super().__init__()

        dim = int(round(base_dim * width_mult))

        # Simple stochastic depth schedule over 3 gated blocks
        dprs = [drop_connect_rate * i / max(1, (3 - 1)) for i in range(3)]

        # Stem (not gated)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU()
        )

        # 3 gated ConvNeXtV2-style blocks
        self.block1 = GatedConvNeXtV2Block(dim, mlp_ratio=mlp_ratio,
                                           kernel_size=kernel_size, drop_path=dprs[0])
        self.block2 = GatedConvNeXtV2Block(dim, mlp_ratio=mlp_ratio,
                                           kernel_size=kernel_size, drop_path=dprs[1])
        self.block3 = GatedConvNeXtV2Block(dim, mlp_ratio=mlp_ratio,
                                           kernel_size=kernel_size, drop_path=dprs[2])

        # Final head conv (NOT gated)
        self.head = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, mask=None):
        """
        Inpainting fine-tuning:
            y = model(masked_rgb)                  # mask=None

        FCMAE-style pretraining (optional):
            y = model(masked_rgb, mask=visible_mask)
            # then compute loss only over masked pixels outside this function.
        """
        x = self.stem(x)
        x = self.block1(x, mask=mask)
        x = self.block2(x, mask=mask)
        x = self.block3(x, mask=mask)
        x = self.head(x)   # (B, 3, H, W)
        return x
