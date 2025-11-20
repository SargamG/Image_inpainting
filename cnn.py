import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        # feature conv
        self.conv_feat = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        # gating conv (outputs gate map)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()  # can use ReLU or ELU

    def forward(self, x):
        feat = self.conv_feat(x)
        gate = self.conv_mask(x)
        gate = torch.sigmoid(gate)
        out = self.activation(self.bn(feat)) * gate
        return out

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.project1 = nn.Sequential(nn.Conv2d(259, 256, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(256),
                                      nn.ELU())
        self.project2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(128),
                                      nn.ELU())
        self.project3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(64),
                                      nn.ELU())
        self.conv1 = GatedConv2d(256, 128, kernel_size=7, stride=1, padding=3)
        self.conv2 = GatedConv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = GatedConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, ae_ft):
        H, W = x.shape[2], x.shape[3]
        ae_ft1 = F.interpolate(ae_ft[0], size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([x, ae_ft1], dim=1)
        x1 = self.project1(x)
        ft1 = self.conv1(x1)
    
        ae_ft2 = F.interpolate(ae_ft[1], size=(H, W), mode='bilinear', align_corners=False)
        ft1 = torch.cat([ft1, ae_ft2], dim=1)
        ft1 = self.project2(ft1)
        ft2 = self.conv2(ft1)
    
        ae_ft3 = F.interpolate(ae_ft[2], size=(H, W), mode='bilinear', align_corners=False)
        ft2 = torch.cat([ft2, ae_ft3], dim=1)
        ft2 = self.project3(ft2)
        ft3 = self.conv3(ft2)
    
        out = self.conv4(ft3)
        return out
    
