"""
PyTorch U-Net mirroring the TensorFlow baseline architecture.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn


class BNConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.conv(x)
        return self.act(x)


class BNUpsampleConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.conv_t(x)
        return self.act(x)


class UNet(nn.Module):
    """
    U-Net with constant filter sizes, matching tensorflow_baseline/framework/model.py.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        num_layers: int = 2,
        base_filters: int = 64,
        upconv_filters: int = 96,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.base_filters = base_filters
        self.upconv_filters = upconv_filters

        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)
        up_kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)

        self.initial_conv = nn.Conv2d(in_channels, base_filters, **conv_kwargs)
        self.c1 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.c1b = BNConvReLU(base_filters, base_filters, **conv_kwargs)

        self.down_blocks: List[BNConvReLU] = nn.ModuleList()
        self.down_blocks_extra: List[BNConvReLU] = nn.ModuleList()
        for _ in range(num_layers):
            self.down_blocks.append(BNConvReLU(base_filters, base_filters, **conv_kwargs))
            self.down_blocks_extra.append(BNConvReLU(base_filters, base_filters, **conv_kwargs))
        self.down_post: List[BNConvReLU] = nn.ModuleList([BNConvReLU(base_filters, base_filters, **conv_kwargs) for _ in range(num_layers)])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.bottleneck1 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.bottleneck2 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.bottleneck_up = BNUpsampleConvReLU(base_filters, base_filters, **up_kwargs)

        self.up_blocks: List[nn.Module] = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict(
                {
                    "conv1": BNConvReLU(base_filters + base_filters, upconv_filters, **conv_kwargs),
                    "conv2": BNConvReLU(upconv_filters, base_filters, **conv_kwargs),
                    "up": BNUpsampleConvReLU(base_filters, base_filters, **up_kwargs),
                }
            )
            self.up_blocks.append(block)

        self.final_conv1 = BNConvReLU(base_filters + base_filters, upconv_filters, **conv_kwargs)
        self.final_conv2 = BNConvReLU(upconv_filters, base_filters, **conv_kwargs)
        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1, stride=1, padding=0)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        c1 = self.c1(x)
        x = self.c1b(c1)
        x = self.pool(x)

        down_features: List[torch.Tensor] = []
        for down1, down_post in zip(self.down_blocks, self.down_post):
            x = down1(x)
            x = down_post(x)
            down_features.append(x)
            x = self.down_blocks_extra[len(down_features) - 1](x)
            x = self.pool(x)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck_up(x)

        for skip, up_block in zip(reversed(down_features), self.up_blocks):
            x = torch.cat([x, skip], dim=1)
            x = up_block["conv1"](x)
            x = up_block["conv2"](x)
            x = up_block["up"](x)

        x = torch.cat([x, c1], dim=1)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return self.classifier(x)

