"""
PyTorch ResUNet sans deep supervision Version1
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)
    

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)
    




class UNet(nn.Module): 
    """
    ResUNet 
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        num_layers: int = 2,
        base_filters: int = 64,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ResBlock(in_channels, base_filters)
        self.enc2 = ResBlock(base_filters, base_filters)
        self.enc3 = ResBlock(base_filters, base_filters)

        # Bottleneck
        self.bottleneck = ResBlock(base_filters, base_filters)

        # Decoder
        self.up2 = UpBlock(base_filters, base_filters)
        self.up1 = UpBlock(base_filters, base_filters)

        # Classifier
        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x)
        x = self.pool(c1)

        c2 = self.enc2(x)
        x = self.pool(c2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up2(x, c2)
        x = self.up1(x, c1)

        return self.classifier(x)