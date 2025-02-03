import torch
import torch.nn as nn

from spmn.asr.adaptive_rotated_conv import AdaptiveRotatedConv2d
from spmn.asr.routing_function import RountingFunction


class ImgEncoder(nn.Module):
    def __init__(self, img_dim: int, memory_width: int):
        super(ImgEncoder, self).__init__()

        self.p_l = nn.Linear(img_dim, memory_width, bias=True)
        self.p_r = nn.Linear(img_dim, memory_width, bias=True)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        """

        :param x: [1, img_dim]
        :return:  [1, memory_width, memory_width]
        """

        x_l = self.p_l(x)  # [1, memory_width]
        x_r = self.p_r(x)  # [1, memory_width]

        x = torch.bmm(torch.transpose(x_l, -2, -1), x_r)  # [1, memory_width, memory_width]

        x = self.conv_layers(x)

        return x


class TextEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 memory_width: int,
                 bottleneck: int = 64
                 ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, memory_width)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, memory_width)
        )

    def forward(self, x):
        x1 = self.fc1(x).unsqueeze(dim=-1)  # [bs, memory_width, 1]
        x2 = self.fc2(x).unsqueeze(dim=-1)  # [bs, memory_width, 1]

        x = torch.bmm(x1, torch.transpose(x2, -2, -1)).unsqueeze(dim=1)  # [bs, 1, memory_width, memory_width]

        return x
