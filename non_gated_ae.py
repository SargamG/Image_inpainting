import torch
import torch.nn as nn

class NormalAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))   # 128->64
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True)) # 64->32
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True))# 32->16
        self.enc4 = nn.Sequential(nn.Conv2d(256, latent_dim, 4, 2, 1), nn.BatchNorm2d(latent_dim), nn.ReLU(True)) #16->8

        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True)) #8->16
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True)) #16->32
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))   #32->64
        self.dec4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)  #64->128
        self.out_act = nn.Sigmoid()

    def encode(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        z = self.enc4(x3)
        return x1, x2, z

    def decode(self, z):
        x = self.dec1(z)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return self.out_act(x)

    def forward(self, x):
        x1, x2, z = self.encode(x)
        out = self.decode(z)
        return out, x1, x2, z
