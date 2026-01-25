import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ResNetEncoder
from model.aspp import ASPP
from model.attention import CBAM
from model.decoder import Decoder


class NeuroSeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = ResNetEncoder()

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.attention = CBAM(channels=256)

        self.decoder = Decoder(low_level_channels=256, out_channels=256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Handle grayscale input (1 → 3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x0, x1, x2, x3, x4 = self.encoder(x)

        x = self.aspp(x4)
        x = self.attention(x)

        x = self.decoder(x, x1)

        x = F.interpolate(
            x, scale_factor=4, mode="bilinear", align_corners=False
        )

        x = self.classifier(x)

        return x
