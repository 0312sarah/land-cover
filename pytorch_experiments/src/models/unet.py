import torch
import torch.nn as nn


class BNConvReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BNUpsampleConvReLU(nn.Module):
    """
    ConvTranspose2d version of "BN -> UpConv -> ReLU"
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, output_padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetTFStyle(nn.Module):
    """
    PyTorch adaptation of the TensorFlow baseline U-Net you showed:
    - first Conv2d (no BN)
    - then BN->Conv->ReLU blocks
    - MaxPool down
    - ConvTranspose up
    - skip connections (concat)
    - final 1x1 conv to num_classes
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        num_layers: int = 4,
        base_channels: int = 64,
        upconv_filters: int = 96,
    ):
        super().__init__()

        filters = base_channels

        # Initial block: Conv then BNConvReLU twice like TF
        self.first_conv = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.first_relu = nn.ReLU(inplace=True)

        self.c1_bnconv = BNConvReLU(filters, filters)
        self.c1_bnconv2 = BNConvReLU(filters, filters)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path layers (store skip tensors)
        self.down_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                "bn1": BNConvReLU(filters, filters),
                "bn2": BNConvReLU(filters, filters),
                "bn3": BNConvReLU(filters, filters),
            })
            self.down_blocks.append(block)

        # Bottleneck
        self.bottleneck_bn1 = BNConvReLU(filters, filters)
        self.bottleneck_bn2 = BNConvReLU(filters, filters)
        self.bottleneck_up = BNUpsampleConvReLU(filters, filters)

        # Up path layers
        self.up_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                "bn_upconv_filters": BNConvReLU(filters + filters, upconv_filters),  # concat doubles channels
                "bn_filters": BNConvReLU(upconv_filters, filters),
                "up": BNUpsampleConvReLU(filters, filters),
            })
            self.up_blocks.append(block)

        # Final up block with c1 skip
        self.final_bn_upconv_filters = BNConvReLU(filters + filters, upconv_filters)
        self.final_bn_filters = BNConvReLU(upconv_filters, filters)

        # Output 1x1 conv -> logits
        self.head = nn.Conv2d(filters, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.first_relu(self.first_conv(x))

        c1 = self.c1_bnconv(x)
        x = self.c1_bnconv2(c1)
        x = self.pool(x)

        skips = []
        for block in self.down_blocks:
            x = block["bn1"](x)
            x = block["bn2"](x)
            skips.append(x)        # like TF down_layers.append(x)
            x = block["bn3"](x)
            x = self.pool(x)

        x = self.bottleneck_bn1(x)
        x = self.bottleneck_bn2(x)
        x = self.bottleneck_up(x)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = torch.cat([x, skip], dim=1)  # concat channels
            x = block["bn_upconv_filters"](x)
            x = block["bn_filters"](x)
            x = block["up"](x)

        x = torch.cat([x, c1], dim=1)
        x = self.final_bn_upconv_filters(x)
        x = self.final_bn_filters(x)

        logits = self.head(x)
        return logits
