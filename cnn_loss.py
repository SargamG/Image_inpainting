import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()

        # Load full VGG once
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg_pretrained))  # single copy
        self.vgg.eval()

        # Map VGG layers to indices
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
        }

        self.selected_indices = [self.layer_map[l] for l in layers]

        # Freeze weights to save memory
        for p in self.vgg.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self, pred, gt):

        # Normalize input to VGG standard
        pred = (pred - self.mean) / self.std
        gt   = (gt - self.mean) / self.std

        total_loss = 0.0
        x_pred = pred
        x_gt = gt

        # RUN THROUGH VGG ONLY ONCE and collect intermediate outputs
        features_pred = []
        features_gt = []

        for i, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_gt = layer(x_gt)

            if i in self.selected_indices:
                features_pred.append(x_pred)
                features_gt.append(x_gt)

                # optional: save memory by detaching unused layers early
                # but we don't need grads anyway

        # Compute perceptual loss
        for fp, fg in zip(features_pred, features_gt):
            total_loss += F.l1_loss(fp, fg)

        return total_loss

def cnn_loss(pred, gt, wt=0.2):
    l1_loss = F.l1_loss(pred, gt)
    perceptual_loss = PerceptualLoss()(pred, gt)
    total_loss = l1_loss + wt*perceptual_loss
    return total_loss
