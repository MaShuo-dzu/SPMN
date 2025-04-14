# encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        self.se_ratio = se_ratio
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True)
        self.swish = nn.SiLU()
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.swish(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.use_res_connection = stride == 1 and in_channels == out_channels

        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(expanded_channels)
            self.swish0 = nn.SiLU()
        else:
            self.expand_conv = None

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels if expand_ratio != 1 else in_channels,
            expanded_channels if expand_ratio != 1 else in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expanded_channels if expand_ratio != 1 else in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels if expand_ratio != 1 else in_channels)
        self.swish1 = nn.SiLU()

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(expanded_channels if expand_ratio != 1 else in_channels,
                                    se_ratio) if se_ratio else None

        # Output phase
        self.project_conv = nn.Conv2d(
            expanded_channels if expand_ratio != 1 else in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish0(x)

        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.project_conv(x)
        x = self.bn2(x)

        if self.use_res_connection:
            x += residual

        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args, final_channel, dropout_rate=0.2):
        super().__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.swish = nn.SiLU()

        # Building MBConv blocks
        self.blocks = nn.ModuleList()
        current_in_channels = 32

        for block_args in blocks_args:
            for i in range(block_args['num_repeat']):
                stride = block_args['stride'] if i == 0 else 1
                in_channels = block_args['in_channels'] if i == 0 else block_args['out_channels']

                block = MBConvBlock(
                    in_channels=in_channels,
                    out_channels=block_args['out_channels'],
                    kernel_size=block_args['kernel_size'],
                    stride=stride,
                    expand_ratio=block_args['expand_ratio'],
                    se_ratio=block_args['se_ratio']
                )
                self.blocks.append(block)
                current_in_channels = block_args['out_channels']

        # Final layers
        self.final_conv = nn.Conv2d(current_in_channels, final_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(final_channel)
        self.dropout = nn.Dropout(dropout_rate)

        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn0(x)
        x = self.swish(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        x = self.dropout(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def efficientnet_b0(final_channel):
    blocks_args = [
        {'num_repeat': 1, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 1, 'in_channels': 32, 'out_channels': 16,
         'se_ratio': 0.25},
        {'num_repeat': 2, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 6, 'in_channels': 16, 'out_channels': 24,
         'se_ratio': 0.25},
        {'num_repeat': 2, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'in_channels': 24, 'out_channels': 40,
         'se_ratio': 0.25},
        {'num_repeat': 3, 'kernel_size': 3, 'stride': 2, 'expand_ratio': 6, 'in_channels': 40, 'out_channels': 80,
         'se_ratio': 0.25},
        {'num_repeat': 3, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'in_channels': 80, 'out_channels': 112,
         'se_ratio': 0.25},
        {'num_repeat': 4, 'kernel_size': 5, 'stride': 2, 'expand_ratio': 6, 'in_channels': 112, 'out_channels': 192,
         'se_ratio': 0.25},
        {'num_repeat': 1, 'kernel_size': 3, 'stride': 1, 'expand_ratio': 6, 'in_channels': 192, 'out_channels': 320,
         'se_ratio': 0.25},
    ]
    return EfficientNet(blocks_args, final_channel)


def test_output(memory_with, backbone):
    with torch.no_grad():
        input = torch.randn(2, 1, memory_with, memory_with)
        output = backbone(input)

    return output.shape[-1]


# 测试代码
if __name__ == "__main__":
    backbone = efficientnet_b0(1)
    print(test_output(500, backbone))
