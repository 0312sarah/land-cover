
# ResUNet sans deep supervision version2
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with optional skip conv if in_channels != out_channels"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class DownBlock(nn.Module):
    """Downsample with conv stride=2 + residual block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.res = ResBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """Upsample with ConvTranspose2d + residual block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.res = ResBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class UNet(nn.Module):
    """ResUNet with optional deep supervision"""
    def __init__(self, in_channels=4, num_classes=10, base_filters=64):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock(in_channels, base_filters)
        self.enc2 = DownBlock(base_filters, base_filters*2)
        self.enc3 = DownBlock(base_filters*2, base_filters*4)
        self.enc4 = DownBlock(base_filters*4, base_filters*8)

        # Bottleneck
        self.bottleneck = ResBlock(base_filters*8, base_filters*16)

        # Decoder
        self.up4 = UpBlock(base_filters*16, base_filters*8)
        self.up3 = UpBlock(base_filters*8, base_filters*4)
        self.up2 = UpBlock(base_filters*4, base_filters*2)
        self.up1 = UpBlock(base_filters*2, base_filters)

        # Classifier
        self.classifier = nn.Conv2d(base_filters, num_classes, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)

        # Bottleneck
        b = self.bottleneck(c4)

        # Decoder
        d4 = self.up4(b, c4)
        d3 = self.up3(d4, c3)
        d2 = self.up2(d3, c2)
        d1 = self.up1(d2, c1)

        return self.classifier(d1)