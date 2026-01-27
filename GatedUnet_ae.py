# ---- Fixed GatedConv2d, GatedUNet and train helper (replace previous definitions) ----
import os, shutil, math
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

class GatedConv2d_UNet(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, activation=True, use_separable=False):
        super().__init__()
        self.use_separable = use_separable
        if use_separable:
            self.feature = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True) if activation else nn.Identity()
            )
            self.gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True),
                nn.Sigmoid()
            )
        else:
            self.feature = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True) if activation else nn.Identity()
            )
            self.gate = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=True),
                nn.Sigmoid()
            )

    def forward(self, x):
        f = self.feature(x)
        g = self.gate(x)
        return f * g


class GatedUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, depth=4, latent_ch=256, use_separable=False):
        super().__init__()
        self.depth = depth
        # encoder channel list (e.g. [32,64,128,256])
        enc_chs = [base_ch * (2**i) for i in range(depth)]

        # Encoder: gated convs with stride=2 (downsampling)
        self.encs = nn.ModuleList()
        prev = in_ch
        for c in enc_chs:
            self.encs.append(GatedConv2d_UNet(prev, c, kernel=3, stride=2, padding=1,
                                         activation=True, use_separable=use_separable))
            prev = c

        # Bottleneck (no downsample)
        self.bottleneck = GatedConv2d_UNet(prev, latent_ch, kernel=3, stride=1, padding=1,
                                      activation=True, use_separable=use_separable)

        # Decoder:
        # decs[0] = first decoder block (NO upsample) -> aligns with enc_chs[-1] (8x8)
        # subsequent decs perform upsample -> gated conv with concat(skip)
        self.decs = nn.ModuleList()

        # First decoder block: no upsample (identity), gated conv takes concat(bottleneck, skip_enc4)
        # in_channels = latent_ch (bottleneck) + enc_chs[-1] (skip)
        self.decs.append(nn.ModuleDict({
            "up": nn.Identity(),
            "gated": GatedConv2d_UNet(in_ch=latent_ch + enc_chs[-1], out_ch=enc_chs[-1],
                                 kernel=3, stride=1, padding=1, activation=True, use_separable=use_separable)
        }))

        # Remaining decoder blocks: upsample then gated conv with the matching skip
        # We'll iterate through enc_chs reversed, skipping the last one (handled above)
        prev_ch = enc_chs[-1]  # current feature channels after first gated_out
        for idx in range(1, len(enc_chs)):
            # target skip channel (one level up)
            skip_ch = enc_chs[-1 - idx]  # enc3, enc2, enc1 ...
            # upsample from prev_ch -> skip_ch (so shapes match after up)
            self.decs.append(nn.ModuleDict({
                "up": nn.ConvTranspose2d(prev_ch, skip_ch, kernel_size=4, stride=2, padding=1),
                "gated": GatedConv2d_UNet(in_ch=skip_ch*2, out_ch=skip_ch,
                                     kernel=3, stride=1, padding=1, activation=True, use_separable=use_separable)
            }))
            prev_ch = skip_ch

        # After last gated_out we are at enc_chs[0] spatial (64x64).
        # Final upsample to restore full resolution 128x128:
        self.final_up = nn.ConvTranspose2d(prev_ch, prev_ch // 2 if (prev_ch // 2) > 0 else prev_ch,
                                           kernel_size=4, stride=2, padding=1)
        final_mid_ch = prev_ch // 2 if (prev_ch // 2) > 0 else prev_ch
        self.final_conv = nn.Conv2d(final_mid_ch, 3, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        skips = []
        out = x
        # encode
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        # bottleneck
        latent = self.bottleneck(out)

        out = latent

        # # decoder: first block (no upsample) uses last skip
        # dec0 = self.decs[0]
        # # concatenate bottleneck + last skip
        # skip = skips[-1]
        # if out.shape[-2:] != skip.shape[-2:]:
        #     out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        # out = torch.cat([out, skip], dim=1)
        # out = dec0["gated"](out)

        # # remaining decoder blocks
        # for i in range(1, len(self.decs)):
        #     dec = self.decs[i]
        #     out = dec["up"](out)                     # upsample to next skip size
        #     skip = skips[-1 - i]                     # corresponding skip
        #     if out.shape[-2:] != skip.shape[-2:]:
        #         out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        #     out = torch.cat([out, skip], dim=1)
        #     out = dec["gated"](out)

        # # final upsample to original resolution
        # out = self.final_up(out)
        # out = self.final_conv(out)
        # out = self.out_act(out)

        # 🔹 RETURN MULTI-SCALE EMBEDDINGS
        x1 = skips[0]   # highest resolution
        x2 = skips[1]   # mid resolution

        return out, latent, x2, x1
