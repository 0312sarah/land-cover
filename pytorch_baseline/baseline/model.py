"""
PyTorch U-Net mirroring the TensorFlow baseline architecture.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F


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
    U-Net encoder + FPN-style multi-scale fusion. Output shape unchanged : [B, num_classes, H, W]
    with constant filter sizes, matching tensorflow_baseline/framework/model.py.
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
        #self.num_layers = num_layers
        #self.base_filters = base_filters
        #self.upconv_filters = upconv_filters

        conv_kwargs = dict(kernel_size=3, stride=1, padding=1)
        up_kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)

        #Encoder
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
        
        
        #Bottleneck
        self.bottleneck1 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.bottleneck2 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        #self.bottleneck_up = BNUpsampleConvReLU(base_filters, base_filters, **up_kwargs)

        #self.up_blocks: List[nn.Module] = nn.ModuleList()
        #for _ in range(num_layers):
            #block = nn.ModuleDict(
            #    {
            #        "conv1": BNConvReLU(base_filters + base_filters, upconv_filters, **conv_kwargs),
            #        "conv2": BNConvReLU(upconv_filters, base_filters, **conv_kwargs),
            #        "up": BNUpsampleConvReLU(base_filters, base_filters, **up_kwargs),
            #   }
            #)
            #self.up_blocks.append(block)

        #self.final_conv1 = BNConvReLU(base_filters + base_filters, upconv_filters, **conv_kwargs)
        #self.final_conv2 = BNConvReLU(upconv_filters, base_filters, **conv_kwargs)
        #self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1, stride=1, padding=0)

        #remplacé par:
        #FPN lateral 1×1 convs 
        self.lateral_c1 = nn.Conv2d(base_filters, base_filters, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(base_filters, base_filters, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(base_filters, base_filters, kernel_size=1)

        #FPN smoothing 
        self.fpn_smooth1 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.fpn_smooth2 = BNConvReLU(base_filters, base_filters, **conv_kwargs)
        self.fpn_smooth3 = BNConvReLU(base_filters, base_filters, **conv_kwargs)

        #Final classifier
        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)
        #Fin remplacé


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
        #for down1, down_post in zip(self.down_blocks, self.down_post):
        for down1, down_post, down_extra in zip(self.down_blocks, self.down_post, self.down_blocks_extra): # remplace ci-dessus, pas utile si rien ne change dans down_blocks_extra
            x = down1(x)
            x = down_post(x)
            down_features.append(x)
            #x = self.down_blocks_extra[len(down_features) - 1](x)
            x = down_extra(x) # remplace ci-dessus, pas utile si rien ne change dans down_blocks_extra
            x = self.pool(x)


        # rajouté : c2, c3 for FPN
        c2 = down_features[0]          # H/2
        c3 = down_features[1]          # H/4
        # fin rajouté

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        #x = self.bottleneck_up(x)

        #Rajouté : FPN top-down 
        p3 = self.lateral_c3(c3)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.lateral_c1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        p3 = self.fpn_smooth3(p3)
        p2 = self.fpn_smooth2(p2)
        p1 = self.fpn_smooth1(p1)

        #Fuse all scales to H 
        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="nearest")
        p3_up = F.interpolate(p3, size=p1.shape[-2:], mode="nearest")

        fused = p1 + p2_up + p3_up

        return self.classifier(fused)
        #Fin rajouté 

        #for skip, up_block in zip(reversed(down_features), self.up_blocks):
        #    x = torch.cat([x, skip], dim=1)
        #    x = up_block["conv1"](x)
        #    x = up_block["conv2"](x)
        #    x = up_block["up"](x)

        #x = torch.cat([x, c1], dim=1)
        #x = self.final_conv1(x)
        #x = self.final_conv2(x)
        #return self.classifier(x)

