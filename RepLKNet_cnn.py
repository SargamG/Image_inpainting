import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Helper: fuse Conv+BN into Conv (for re-parameterization) ----------

def fuse_conv_bn(conv, bn):
    """
    Fuse a Conv2d and BatchNorm2d into a single Conv2d (kernel + bias).
    Assumes conv.bias is None or zero.
    """
    if conv is None:
        # no conv branch
        raise ValueError("conv is None in fuse_conv_bn")

    w = conv.weight            # (C_out, C_in/groups, kH, kW)
    if conv.bias is None:
        bias = torch.zeros(w.size(0), device=w.device)
    else:
        bias = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
    b_fused = beta + (bias - mean) * gamma / std
    return w_fused, b_fused


# ---------- Large depthwise conv with structural re-parameterization ----------

class RepLKDepthwise(nn.Module):
    """
    Large depthwise conv with structural re-parameterization:
      Training:  large DW k_large + small DW k_small + identity BN
      Deploy:    single DW conv with fused large kernel

    This follows the RepLKNet guideline of re-parameterizing small
    kernels into a large one to ease optimization. :contentReference[oaicite:4]{index=4}
    """
    def __init__(self, channels, k_large=13, k_small=3, deploy=False):
        super().__init__()
        self.channels = channels
        self.k_large = k_large
        self.k_small = k_small
        self.deploy = deploy

        if deploy:
            # single depthwise conv after fusion
            self.dw_reparam = nn.Conv2d(
                channels, channels,
                kernel_size=k_large,
                padding=k_large // 2,
                groups=channels,
                bias=True,
            )
        else:
            # large depthwise branch
            self.dw_large = nn.Conv2d(
                channels, channels,
                kernel_size=k_large,
                padding=k_large // 2,
                groups=channels,
                bias=False,
            )
            self.bn_large = nn.BatchNorm2d(channels)

            # small depthwise branch
            self.dw_small = nn.Conv2d(
                channels, channels,
                kernel_size=k_small,
                padding=k_small // 2,
                groups=channels,
                bias=False,
            )
            self.bn_small = nn.BatchNorm2d(channels)

            # identity branch (BN only)
            self.bn_identity = nn.BatchNorm2d(channels)

    def forward(self, x):
        if self.deploy:
            return self.dw_reparam(x)

        out = self.bn_large(self.dw_large(x)) \
            + self.bn_small(self.dw_small(x)) \
            + self.bn_identity(x)
        return out

    def switch_to_deploy(self):
        """Fuse large + small + identity branches into a single large DW conv."""
        if self.deploy:
            return

        # fuse each branch to (kernel, bias)
        k_large, b_large = fuse_conv_bn(self.dw_large, self.bn_large)

        k_small, b_small = fuse_conv_bn(self.dw_small, self.bn_small)
        # pad small kernel into large size
        kL = self.k_large
        kS = self.k_small
        pad = (kL - kS) // 2
        k_small_padded = torch.zeros_like(k_large)
        k_small_padded[:, :, pad:pad + kS, pad:pad + kS] = k_small

        # identity branch as a depthwise conv with 1 at center
        k_identity = torch.zeros_like(k_large)
        center = kL // 2
        # for depthwise, weight shape is (C, 1, k, k); put 1 at center
        for c in range(self.channels):
            k_identity[c, 0, center, center] = 1.0
        # fuse BN into identity "conv"
        gamma = self.bn_identity.weight
        beta = self.bn_identity.bias
        mean = self.bn_identity.running_mean
        var = self.bn_identity.running_var
        eps = self.bn_identity.eps
        std = torch.sqrt(var + eps)
        k_identity = k_identity * (gamma / std).reshape(-1, 1, 1, 1)
        b_identity = beta - mean * gamma / std

        # sum kernels and biases
        k_fused = k_large + k_small_padded + k_identity
        b_fused = b_large + b_small + b_identity

        # create the re-parameterized conv
        self.dw_reparam = nn.Conv2d(
            self.channels, self.channels,
            kernel_size=self.k_large,
            padding=self.k_large // 2,
            groups=self.channels,
            bias=True,
        )
        self.dw_reparam.weight.data = k_fused
        self.dw_reparam.bias.data = b_fused

        # clean up training-time branches
        del self.dw_large, self.bn_large
        del self.dw_small, self.bn_small
        del self.bn_identity
        self.deploy = True


# ---------- Gated RepLK block (large DW conv + gate + shortcut) ----------

class GatedRepLKBlock(nn.Module):
    """
    RepLKNet-style large kernel block:
      x -> (RepLKDepthwise large DW) -> gated -> 1x1 conv -> + identity

    - Very large depthwise kernel (e.g. 13x13) (Guideline 1). :contentReference[oaicite:5]{index=5}
    - Identity shortcut (Guideline 2).
    - Structural reparameterization via RepLKDepthwise (Guideline 3).
    - Gated conv: feature * sigmoid(gate).
    """
    def __init__(
        self,
        channels: int,
        k_large: int = 13,
        k_small: int = 3,
        deploy: bool = False,
    ):
        super().__init__()
        self.channels = channels

        # large depthwise conv with structural re-param
        self.dw_replk = RepLKDepthwise(
            channels, k_large=k_large, k_small=k_small, deploy=deploy
        )

        # gate branch: 3x3 depthwise conv
        self.gate_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # pointwise conv + BN + activation
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

        # dropout-style stochastic depth (optional, here omitted for simplicity)

    def forward(self, x):
        identity = x

        feat = self.dw_replk(x)
        gate = torch.sigmoid(self.gate_conv(x))
        out = feat * gate

        out = self.pw(out)
        out = self.bn(out)
        out = self.act(out)

        out = out + identity  # identity shortcut is vital for large kernels
        return out

    def switch_to_deploy(self):
        self.dw_replk.switch_to_deploy()


# ---------- 4-layer RepLKNet-style inpainting CNN ----------

class GatedRepLKNetInpaint4(nn.Module):
    """
    4-block CNN for image inpainting, inspired by RepLKNet.

    Structure:
      - Stem:   3x3 conv (3 -> C), stride 1, keep H,W (not gated).
      - Block1: GatedRepLKBlock (large DW conv + gate + shortcut).
      - Block2: GatedRepLKBlock (gated).
      - Block3: GatedRepLKBlock (gated).
      - Block4: Plain 3x3 Conv2d (C -> 3), stride 1 (NOT gated).

    All convolutions have stride=1 and appropriate padding, so
    output spatial dimensions equal input spatial dimensions.
    """
    def __init__(
        self,
        in_channels: int = 3,   # masked RGB image only
        base_channels: int = 32,
        width_mult: float = 1.0,
        k_large: int = 13,
        k_small: int = 3,
        deploy: bool = False,
    ):
        super().__init__()
        c = int(round(base_channels * width_mult))

        # Stem: simple conv, not counted in "gated 3 blocks"
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )

        # 3 gated RepLK blocks
        self.block1 = GatedRepLKBlock(c, k_large=k_large, k_small=k_small, deploy=deploy)
        self.block2 = GatedRepLKBlock(c, k_large=k_large, k_small=k_small, deploy=deploy)
        self.block3 = GatedRepLKBlock(c, k_large=k_large, k_small=k_small, deploy=deploy)

        # Final head conv: NOT gated
        self.head = nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1, bias=True)

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

    def switch_to_deploy(self):
        """Fuse large + small + identity branches in all blocks for inference."""
        self.block1.switch_to_deploy()
        self.block2.switch_to_deploy()
        self.block3.switch_to_deploy()
