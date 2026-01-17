import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- LayerNorm in NCHW (simple normalization) --------

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for NCHW tensors.
    HorNet uses LN before g_nConv and FFN. :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W)
        x_perm = x.permute(0, 2, 3, 1)      # (B, H, W, C)
        x_norm = self.ln(x_perm)
        return x_norm.permute(0, 3, 1, 2)   # (B, C, H, W)


# -------- MLP-like channel mixing layer (FFN) --------

class ConvFFN(nn.Module):
    """
    MLP-like channel mixing: 1x1 -> GELU -> 1x1.
    Mirrors the FFN sub-layer in HorNet blocks. :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# -------- Multi-head recursive gated convolution (HorNet-style gₙConv) --------

class MultiHeadGNConv(nn.Module):
    """
    Simplified HorNet g_nConv with multi-head gating. :contentReference[oaicite:3]{index=3}

    - Recursive gating / high-order spatial interactions via channel splits.
    - Depthwise spatial conv for input-adaptive spatial mixing.
    - Multi-head gating: channels are split into heads when multiplying gates.
    """
    def __init__(
        self,
        dim: int,
        order: int = 3,         # interaction order n
        kernel_size: int = 7,   # depthwise spatial conv
        num_heads: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.order = order
        self.num_heads = num_heads

        # dims for each recursion order, like HorNet (C, C/2, C/4, ...) :contentReference[oaicite:4]{index=4}
        dims = [dim // (2 ** i) for i in range(order)]
        dims.reverse()    # smallest first, like paper code
        self.dims = dims  # e.g. for dim=64, order=3 -> [16, 32, 64]

        # Input projection: C -> 2C
        self.proj_in = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=True)

        # Depthwise spatial conv over concatenated q_k
        dw_channels = sum(self.dims)
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=dw_channels,
            bias=True,
        )

        # Order-wise 1x1 projections g_k (MLP-like mixing between orders)
        projs = []
        for i in range(order - 1):
            projs.append(nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=1, bias=True))
        self.projs = nn.ModuleList(projs)

        # Output projection: C -> C
        self.proj_out = nn.Conv2d(self.dims[-1], dim, kernel_size=1, bias=True)

    def _head_gate(self, a, b):
        """
        Multi-head gating: split channels into num_heads groups and
        apply elementwise gating within each head.
        """
        B, C, H, W = a.shape
        assert C % self.num_heads == 0
        hdim = C // self.num_heads
        a = a.view(B, self.num_heads, hdim, H, W)
        b = b.view(B, self.num_heads, hdim, H, W)
        out = a * b
        return out.view(B, C, H, W)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        # 1) Linear projection: C -> 2C
        x_proj = self.proj_in(x)

        # 2) Split first part (y) and concatenated q_k
        y, q_all = torch.split(x_proj, (self.dims[0], sum(self.dims)), dim=1)

        # 3) Depthwise spatial conv over all q_k
        q_all = self.dwconv(q_all)

        # 4) Split back into list of q_k (per order)
        q_list = torch.split(q_all, self.dims, dim=1)

        # 5) Recursive multi-head gating
        #    p0 = y * q0   (head-wise)
        p = self._head_gate(y, q_list[0])

        #    pk+1 = proj_k(pk) * q_{k+1}
        for i in range(self.order - 1):
            p = self.projs[i](p)
            p = self._head_gate(p, q_list[i + 1])

        # 6) Output projection
        out = self.proj_out(p)
        return out


# -------- HorNet-style block with residual + simple norm + gₙConv + FFN --------

class HorNetStyleBlock(nn.Module):
    """
    One HorNet-like block for our 4-layer CNN:

      x -> LN -> gₙConv -> x + ... (residual 1)
      -> LN -> FFN -> x + ...       (residual 2)

    This block is *gated* and used for the first 3 layers.
    """
    def __init__(
        self,
        dim: int,
        order: int = 3,
        kernel_size: int = 7,
        num_heads: int = 4,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.gnconv = MultiHeadGNConv(
            dim=dim,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
        )

        self.norm2 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim, expansion=ffn_expansion)

    def forward(self, x):
        # Spatial high-order interactions with recursive multi-head gated conv
        x_res = x
        x = self.norm1(x)
        x = self.gnconv(x)
        x = x_res + x       # residual 1

        # MLP-like channel mixing
        x_res2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x_res2 + x      # residual 2
        return x

class HorNetInpaint4(nn.Module):
    """
    4-layer HorNet-inspired CNN for image inpainting.

    - First 3 blocks: gated HorNetStyleBlock (recursive gated conv, multi-head, FFN).
    - Last block: plain 3x3 conv C -> 3 (RGB).
    - No downsampling: input and output spatial sizes are equal.
    """
    def __init__(
        self,
        in_channels: int = 3,   # masked RGB image
        base_channels: int = 32,
        width_mult: float = 1.0,
        order: int = 3,
        kernel_size: int = 7,
        num_heads: int = 4,
        ffn_expansion: int = 4,
    ):
        super().__init__()
        C = int(round(base_channels * width_mult))

        # Stem (NOT gated)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        # 3 HorNet-style gated blocks
        self.block1 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )
        self.block2 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )
        self.block3 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )

        # Head: plain 3x3 conv (NOT gated)
        self.head = nn.Conv2d(C, 3, kernel_size=3, stride=1, padding=1, bias=True)

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

class HorNetInpaint4_WithLatent(nn.Module):
    """
    HorNet-based CNN conditioned on AE latent embedding.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 256,   # must match AE latent channels
        base_channels: int = 32,
        width_mult: float = 1.0,
        order: int = 3,
        kernel_size: int = 7,
        num_heads: int = 4,
        ffn_expansion: int = 4,
    ):
        super().__init__()

        C = int(round(base_channels * width_mult))

        # -------- Latent fusion --------
        # After concat(image, latent_up) → project back to 3 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + latent_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

        # -------- Stem --------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
        )

        # -------- HorNet blocks --------
        self.block1 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )
        self.block2 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )
        self.block3 = HorNetStyleBlock(
            dim=C,
            order=order,
            kernel_size=kernel_size,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
        )

        # -------- Output head --------
        self.head = nn.Conv2d(C, 3, kernel_size=3, padding=1)

    def forward(self, masked_img, latent):
        """
        masked_img: (B, 3, H, W)
        latent:     (B, L, h, w)
        """

        #1. Upsample latent to image resolution
        latent_up = F.interpolate(
            latent,
            size=masked_img.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        #2. Concatenate image + latent
        x = torch.cat([masked_img, latent_up], dim=1)

        #3. Fuse via 1×1 conv
        x = self.fusion(x)

        #4. HorNet CNN
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)

        return x

