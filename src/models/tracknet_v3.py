from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BasicConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MultiScaleDownBlock(nn.Module):
    """TrackNet encoder block: 1x1 / 3x3 / 5x5 branches + residual fusion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = BasicConv(in_channels, out_channels, 1)
        self.conv_2 = BasicConv(out_channels, out_channels, 3)

        self.conv_3 = BasicConv(in_channels, out_channels, 3)
        self.conv_4 = BasicConv(out_channels, out_channels, 3)

        self.conv_5 = BasicConv(in_channels, out_channels, 5)
        self.conv_6 = BasicConv(out_channels, out_channels, 3)

        self.conv_7 = BasicConv(out_channels * 3, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_1 = self.conv_2(self.conv_1(x))
        branch_2 = self.conv_4(self.conv_3(x))
        branch_3 = self.conv_6(self.conv_5(x))
        fused = torch.cat([branch_1, branch_2, branch_3], dim=1)
        return self.conv_7(fused) + branch_2


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv_1 = BasicConv(in_channels, hidden_channels, 3)
        self.conv_2 = BasicConv(hidden_channels, hidden_channels, 3)
        self.conv_3 = BasicConv(hidden_channels, hidden_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = BasicConv(in_channels, out_channels, 3)
        self.conv_2 = BasicConv(out_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_MLP(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_MLP(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv2d(attention)
        return self.sigmoid(attention)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x) * x
        # Official implementation only uses channel attention in forward.
        # Keep spatial_attention module so pretrained weights still load cleanly.
        return x


class TrackNetV2(nn.Module):
    """TrackNetV2-style encoder-decoder used by the provided checkpoint."""

    def __init__(self, in_channels: int = 9, out_channels: int = 3) -> None:
        super().__init__()
        self.down_block_1 = MultiScaleDownBlock(in_channels, 64)
        self.down_block_2 = MultiScaleDownBlock(64, 128)
        self.down_block_3 = MultiScaleDownBlock(128, 256)
        self.bottleneck = BottleneckBlock(256, 512)

        self.up_block_1 = UpBlock(512 + 256, 256)
        self.up_block_2 = UpBlock(256 + 128, 128)
        self.up_block_3 = UpBlock(128 + 64, 64)

        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(64)
        self.cbam0_2 = CBAM(256)
        self.cbam1_2 = CBAM(128)
        self.cbam2_2 = CBAM(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.predictor = nn.Conv2d(64, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_1 = self.down_block_1(x)
        skip_2 = self.down_block_2(self.pool(skip_1))
        skip_3 = self.down_block_3(self.pool(skip_2))

        bottleneck = self.bottleneck(self.pool(skip_3))

        skip_3 = self.cbam0_2(skip_3)
        up_1 = F.interpolate(bottleneck, scale_factor=2, mode="nearest")
        up_1 = self.up_block_1(torch.cat([up_1, skip_3], dim=1))
        up_1 = self.cbam1(up_1)

        skip_2 = self.cbam1_2(skip_2)
        up_2 = F.interpolate(up_1, scale_factor=2, mode="nearest")
        up_2 = self.up_block_2(torch.cat([up_2, skip_2], dim=1))
        up_2 = self.cbam2(up_2)

        skip_1 = self.cbam2_2(skip_1)
        up_3 = F.interpolate(up_2, scale_factor=2, mode="nearest")
        up_3 = self.up_block_3(torch.cat([up_3, skip_1], dim=1))
        up_3 = self.cbam3(up_3)

        return torch.sigmoid(self.predictor(up_3))


# Backward-compatible alias for the rest of the project.
TrackNetV3 = TrackNetV2
