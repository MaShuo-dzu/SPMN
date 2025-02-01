import torch
from torch import nn
import torch.nn.functional as F

from spmn.asr.adaptive_rotated_conv import AdaptiveRotatedConv2d
from spmn.asr.routing_function import RountingFunction


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(x.size(0), x.size(1), 1, 1)
        return x * y


class Compress(nn.Module):
    def __init__(self, in_channels, expansion_factor=6, se_ratio=0.25):
        super(Compress, self).__init__()

        # Expansion layer (pointwise 1x1 convolution)
        self.expand = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1)
        self.expand_bn = nn.BatchNorm2d(in_channels * expansion_factor)

        routing_function_depthwise1 = RountingFunction(in_channels=(in_channels * expansion_factor), kernel_number=4)
        routing_function_depthwise2 = RountingFunction(in_channels=(in_channels * expansion_factor) // 2, kernel_number=4)

        # Depthwise convolution layer
        self.depthwise = nn.Sequential(
            AdaptiveRotatedConv2d(in_channels=(in_channels * expansion_factor), out_channels=(in_channels * expansion_factor) // 2,
                                  kernel_size=3, padding=1, rounting_func=routing_function_depthwise1, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d((in_channels * expansion_factor) // 2),
            nn.ReLU(),
            AdaptiveRotatedConv2d(in_channels=(in_channels * expansion_factor) // 2, out_channels=(in_channels * expansion_factor),
                                  kernel_size=3, padding=1, rounting_func=routing_function_depthwise2, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d((in_channels * expansion_factor)),
            nn.ReLU()
        )

        # Squeeze-and-excitation layer
        self.se = SEBlock(in_channels * expansion_factor, int(in_channels * expansion_factor * se_ratio))

        # Pointwise 1x1 convolution layer
        self.project = nn.Conv2d(in_channels * expansion_factor, 1, kernel_size=1)
        self.project_bn = nn.BatchNorm2d(1)

        # Skip connection (if in_channels == out_channels)
        self.use_residual = (in_channels == 1)

    def forward(self, x):
        residual = x

        # Expansion and depthwise convolutions
        x = F.relu(self.expand_bn(self.expand(x)))
        x = self.depthwise(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Projection
        x = self.project_bn(self.project(x))

        # Add residual connection if possible
        if self.use_residual:
            x += residual

        return x
