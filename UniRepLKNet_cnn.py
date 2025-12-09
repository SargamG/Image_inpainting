import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- SE Block (as in UniRepLKNet guideline 1) ----------

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


# ---------- Simple ConvNeXt-style FFN (1x1 -> GELU -> 1x1) ----------

class FFN(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(channels)  # can be folded at inference

    def forward(self, x):
        out = self.pw1(x)
        out = self.act(out)
        out = self.pw2(out)
        out = self.bn(out)
        return out


# ---------- Dilated Reparam Depthwise Block (UniRepLKNet core) ----------

class DilatedReparamDW(nn.Module):
    """
    UniRepLKNet-style Dilated Reparam Block (train-time) that can be
    collapsed into a single K×K depthwise conv (deploy-time).
    """

    def __init__(
        self,
        channels: int,
        K: int = 13,
        k_list=(5, 7, 3, 3, 3),
        d_list=(1, 2, 3, 4, 5),
        deploy: bool = False
    ):
        super().__init__()
        self.channels = channels
        self.K = K
        self.deploy = deploy
        assert len(k_list) == len(d_list)

        if deploy:
            # Single fused conv for inference
            self.fused_conv = nn.Conv2d(
                channels, channels,
                kernel_size=K,
                padding=K // 2,
                groups=channels,
                bias=True
            )
        else:
            # Main K×K depthwise conv
            self.main = nn.Sequential(
                nn.Conv2d(
                    channels, channels,
                    kernel_size=K,
                    padding=K // 2,
                    groups=channels,
                    bias=False
                ),
                nn.BatchNorm2d(channels)
            )

            # Dilated branches
            branches = []
            for k, d in zip(k_list, d_list):
                pad = d * (k // 2)
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels, channels,
                            kernel_size=k,
                            padding=pad,
                            dilation=d,
                            groups=channels,
                            bias=False
                        ),
                        nn.BatchNorm2d(channels)
                    )
                )
            self.branches = nn.ModuleList(branches)

    # -----------------------------
    #         FUSION UTILS
    # -----------------------------
    @staticmethod
    def fuse_conv_bn(conv, bn):
        """Fuse Conv2d + BatchNorm2d into a single Conv2d kernel & bias."""
        w = conv.weight
        if conv.bias is None:
            b = torch.zeros(w.size(0), device=w.device)
        else:
            b = conv.bias

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta + (b - mean) * gamma / std
        return w_fused, b_fused

    def _expand_dilated_kernel(self, k_small, dilation, K_final):
        """
        Build a K_final × K_final kernel from a smaller dilated kernel.
        Matches Fig. 2 in UniRepLKNet: sparse large kernel. 
        """
        C = k_small.size(0)
        k_small_size = k_small.size(-1)

        big = torch.zeros((C, 1, K_final, K_final), device=k_small.device)

        center = K_final // 2
        small_center = k_small_size // 2

        # Place nonzero elements with spacing = dilation
        for i in range(k_small_size):
            for j in range(k_small_size):
                di = (i - small_center) * dilation
                dj = (j - small_center) * dilation
                big[:, 0, center + di, center + dj] = k_small[:, 0, i, j]

        return big

    # -----------------------------
    #     MAIN SWITCH FUNCTION
    # -----------------------------
    def switch_to_deploy(self):
        """Convert all parallel branches into a single fused K×K depthwise conv."""
        if self.deploy:
            return  # Already in inference mode

        K = self.K
        C = self.channels

        # Initialize empty big kernel & bias
        fused_kernel = torch.zeros((C, 1, K, K), device=self.main[0].weight.device)
        fused_bias = torch.zeros((C,), device=self.main[0].weight.device)

        # --------- Fuse main branch ---------
        Wm, bm = self.fuse_conv_bn(self.main[0], self.main[1])
        fused_kernel += Wm
        fused_bias += bm

        # --------- Fuse dilated branches ---------
        for branch in self.branches:
            conv = branch[0]
            bn = branch[1]

            # fuse conv+bn
            Wb, bb = self.fuse_conv_bn(conv, bn)

            # expand dilated kernel to K×K
            k_small = conv.kernel_size[0]
            dilation = conv.dilation[0]
            expanded = self._expand_dilated_kernel(Wb, dilation, K)

            fused_kernel += expanded
            fused_bias += bb

        # --------- Register final fused conv ---------
        self.fused_conv = nn.Conv2d(
            C, C, kernel_size=K, padding=K // 2, groups=C, bias=True
        ).to(fused_kernel.device)

        self.fused_conv.weight.data = fused_kernel
        self.fused_conv.bias.data = fused_bias

        # Remove training-time modules
        del self.main
        del self.branches
        self.deploy = True

    # -----------------------------
    #         FORWARD
    # -----------------------------
    def forward(self, x):
        if self.deploy:
            return self.fused_conv(x)
        else:
            out = self.main(x)
            for branch in self.branches:
                out = out + branch(x)
            return out


# ---------- Gated UniRepLK-like block (SmaK or LarK) ----------

class GatedUniRepLKBlock(nn.Module):
    """
    UniRepLKNet-inspired block with:
      - a depthwise part (either 3x3 DW or DilatedReparamDW),
      - a gate branch (3x3 DW -> sigmoid),
      - SE block,
      - FFN,
      - identity residual.

    This follows the guidelines:
      - efficient structures (SE + FFN),
      - large kernels via Dilated Reparam Block (LarK),
      - small 3x3 kernels for extra depth (SmaK). :contentReference[oaicite:8]{index=8}
    """
    def __init__(
        self,
        channels: int,
        use_large_kernel: bool = True,
        K: int = 13,
        k_list=(5, 7, 3, 3, 3),
        d_list=(1, 2, 3, 4, 5),
    ):
        super().__init__()

        if use_large_kernel:
            self.dw = DilatedReparamDW(
                channels, K=K, k_list=k_list, d_list=d_list
            )
        else:
            # SmaK-style 3x3 depthwise conv
            self.dw = nn.Sequential(
                nn.Conv2d(
                    channels, channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
            )

        # Gate branch: depthwise 3x3
        self.gate_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # SE + FFN for depth & channel mixing
        self.se = SEBlock(channels, reduction=4)
        self.ffn = FFN(channels, expansion=4)

    def forward(self, x):
        identity = x

        feat = self.dw(x)
        gate = torch.sigmoid(self.gate_conv(x))
        out = feat * gate           # gated spatial features

        out = self.se(out)
        out = self.ffn(out)

        return out + identity       # identity shortcut


# ---------- 4-layer UniRepLKNet-style inpainting CNN ----------

class GatedUniRepLKInpaint4(nn.Module):
    """
    4-block UniRepLKNet-inspired CNN for image inpainting.

    Layout:
      - Stem:   3x3 conv (3 -> C), GELU (NOT gated).
      - Block1: GatedUniRepLKBlock with small 3x3 DW (SmaK-style, gated).
      - Block2: GatedUniRepLKBlock with DilatedReparamDW (LarK-style, gated).
      - Block3: GatedUniRepLKBlock with DilatedReparamDW (LarK-style, gated).
      - Block4: Plain 3x3 Conv2d (C -> 3), no gating.

    All convs have stride=1 and appropriate padding, so spatial
    dimensions are preserved (good for inpainting).
    """
    def __init__(
        self,
        in_channels: int = 3,    # masked RGB input
        base_channels: int = 32,
        width_mult: float = 1.0,
        K_large: int = 13,
    ):
        super().__init__()
        C = int(round(base_channels * width_mult))

        # Stem: simple conv, not gated
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.GELU(),
        )

        # Block1: SmaK-style (3x3 DW), gated
        self.block1 = GatedUniRepLKBlock(
            C,
            use_large_kernel=False,   # 3x3 DW
        )

        # Blocks2–3: LarK-style (Dilated Reparam large kernel), gated
        k_list = (5, 7, 3, 3, 3)
        d_list = (1, 2, 3, 4, 5)     # UniRepLKNet default for K=13 :contentReference[oaicite:9]{index=9}

        self.block2 = GatedUniRepLKBlock(
            C,
            use_large_kernel=True,
            K=K_large,
            k_list=k_list,
            d_list=d_list,
        )
        self.block3 = GatedUniRepLKBlock(
            C,
            use_large_kernel=True,
            K=K_large,
            k_list=k_list,
            d_list=d_list,
        )

        # Head: plain 3x3 conv to RGB (NOT gated)
        self.head = nn.Conv2d(C, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        """
        x: (B, 3, H, W) masked/occluded RGB
        returns: (B, 3, H, W) inpainted RGB
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.head(x)
        return x
