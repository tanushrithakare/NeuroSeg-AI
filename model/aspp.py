import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.conv6 = nn.Conv2d(
            in_channels, out_channels, 3, padding=6, dilation=6, bias=False
        )
        self.conv12 = nn.Conv2d(
            in_channels, out_channels, 3, padding=12, dilation=12, bias=False
        )
        self.conv18 = nn.Conv2d(
            in_channels, out_channels, 3, padding=18, dilation=18, bias=False
        )

        self.project = nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv6(x)
        x3 = self.conv12(x)
        x4 = self.conv18(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.project(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
