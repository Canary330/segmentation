from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(channels * 2, channels),
            DepthwiseSeparableConv(channels, channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MobileUNetFPN(nn.Module):
    """MobileNet encoder + explicit FPN + UNet-style decoder."""

    def __init__(
        self,
        num_classes: int,
        fpn_channels: int = 128,
        dropout: float = 0.1,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = mobilenet_v2(weights=weights).features

        self.stage1 = nn.Sequential(*backbone[:2])   # 1/2
        self.stage2 = nn.Sequential(*backbone[2:4])  # 1/4
        self.stage3 = nn.Sequential(*backbone[4:7])  # 1/8
        self.stage4 = nn.Sequential(*backbone[7:14])  # 1/16
        self.stage5 = nn.Sequential(*backbone[14:19])  # 1/32

        self.lateral1 = ConvBNAct(16, fpn_channels, kernel_size=1)
        self.lateral2 = ConvBNAct(24, fpn_channels, kernel_size=1)
        self.lateral3 = ConvBNAct(32, fpn_channels, kernel_size=1)
        self.lateral4 = ConvBNAct(96, fpn_channels, kernel_size=1)
        self.lateral5 = ConvBNAct(1280, fpn_channels, kernel_size=1)

        self.decoder4 = DecoderBlock(fpn_channels)
        self.decoder3 = DecoderBlock(fpn_channels)
        self.decoder2 = DecoderBlock(fpn_channels)
        self.decoder1 = DecoderBlock(fpn_channels)

        self.head = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        p5 = self.lateral5(s5)
        p4 = self.lateral4(s4) + self._upsample_like(p5, s4)
        p3 = self.lateral3(s3) + self._upsample_like(p4, s3)
        p2 = self.lateral2(s2) + self._upsample_like(p3, s2)
        p1 = self.lateral1(s1) + self._upsample_like(p2, s1)

        d4 = self.decoder4(p5, p4)
        d3 = self.decoder3(d4, p3)
        d2 = self.decoder2(d3, p2)
        d1 = self.decoder1(d2, p1)

        logits = self.head(F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False))
        return logits
