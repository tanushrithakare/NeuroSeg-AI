import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, low_level_channels, out_channels):
        super().__init__()

        self.low_level_conv = nn.Conv2d(
            low_level_channels, 48, kernel_size=1, bias=False
        )
        self.low_level_bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(48 + out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, high_level, low_level):
        low = self.low_level_conv(low_level)
        low = self.low_level_bn(low)
        low = self.relu(low)

        high = F.interpolate(
            high_level, size=low.shape[2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([high, low], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x
