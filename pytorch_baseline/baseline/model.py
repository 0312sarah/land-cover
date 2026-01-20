"""
PyTorch U-Net mirroring the TensorFlow baseline architecture.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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


class UNetClassic(nn.Module):
    """
    Classic U-Net with encoder/decoder feature pyramids.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        features: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feat in features:
            self.downs.append(DoubleConv(channels, feat))
            channels = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        rev_features = list(reversed(features))
        up_channels = features[-1] * 2
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(up_channels, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat * 2, feat))
            up_channels = feat

        self.classifier = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.classifier(x)


class UNetResNet34(nn.Module):
    """
    U-Net decoder on top of a ResNet34 encoder.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        pretrained: bool = True,
        decoder_channels: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 64]

        from torchvision.models import ResNet34_Weights, resnet34

        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = encoder.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if in_channels >= 3:
                    new_conv.weight[:, :3] = old_conv.weight
                    if in_channels > 3:
                        nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
                else:
                    new_conv.weight[:] = old_conv.weight[:, :in_channels]
            encoder.conv1 = new_conv

        self.encoder = encoder
        self.up4 = nn.ConvTranspose2d(512, decoder_channels[0], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(decoder_channels[0] + 256, decoder_channels[0])

        self.up3 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(decoder_channels[1] + 128, decoder_channels[1])

        self.up2 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(decoder_channels[2] + 64, decoder_channels[2])

        self.up1 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(decoder_channels[3] + 64, decoder_channels[3])

        self.up0 = nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], kernel_size=2, stride=2)
        self.dec0 = DoubleConv(decoder_channels[3], decoder_channels[3])

        self.classifier = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x1 = self.encoder.layer1(self.encoder.maxpool(x0))
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        d4 = self.up4(x4)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(d0)
        return self.classifier(d0)


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    num_layers: int = 2,
    base_filters: int = 64,
    upconv_filters: int = 96,
    features: Optional[Sequence[int]] = None,
    pretrained: bool = True,
    decoder_channels: Optional[Sequence[int]] = None,
) -> nn.Module:
    key = str(name).lower()
    if key in ("unet", "unet_baseline"):
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            base_filters=base_filters,
            upconv_filters=upconv_filters,
        )
    if key in ("unet_classic", "unet_standard"):
        return UNetClassic(in_channels=in_channels, num_classes=num_classes, features=features)
    if key in ("unet_resnet34", "unet_resnet34_pretrained"):
        return UNetResNet34(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            decoder_channels=decoder_channels,
        )
    raise ValueError(f"Unknown model name: {name}")

