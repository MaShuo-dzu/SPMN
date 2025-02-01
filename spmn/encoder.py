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
    def __init__(self, seq_len: int, memory_width: int):
        super(TextEncoder, self).__init__()

        self.memory_width = memory_width
        self.seq_len = seq_len

        assert seq_len >= 16, "seq_len 必须>= 16 ！"

        self.p_l = nn.Parameter(torch.randn(seq_len, 1, memory_width // 8))
        self.p_r = nn.Parameter(torch.randn(seq_len, 1, memory_width // 8))

        # memory_width / 8 -> memory_width
        routing_function_11 = RountingFunction(in_channels=seq_len, kernel_number=4)
        routing_function_12 = RountingFunction(in_channels=seq_len // 2,
                                                       kernel_number=4)
        routing_function_21 = RountingFunction(in_channels=seq_len // 2, kernel_number=4)
        routing_function_22 = RountingFunction(in_channels=seq_len // 4,
                                                       kernel_number=4)
        routing_function_31 = RountingFunction(in_channels=seq_len // 4, kernel_number=4)
        routing_function_32 = RountingFunction(in_channels=seq_len // 8,
                                                       kernel_number=4)

        self.conv_layers = nn.Sequential(
            AdaptiveRotatedConv2d(in_channels=seq_len,
                                  out_channels=seq_len // 2,
                                  kernel_size=3, padding=1, rounting_func=routing_function_11, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len // 2),
            nn.ReLU(),
            AdaptiveRotatedConv2d(in_channels=seq_len // 2,
                                  out_channels=seq_len,
                                  kernel_size=3, padding=1, rounting_func=routing_function_12, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len),
            nn.ReLU(),

            nn.ConvTranspose2d(seq_len, seq_len // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(seq_len // 2),
            nn.ReLU(),

            AdaptiveRotatedConv2d(in_channels=seq_len // 2,
                                  out_channels=seq_len // 4,
                                  kernel_size=3, padding=1, rounting_func=routing_function_21, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len // 4),
            nn.ReLU(),
            AdaptiveRotatedConv2d(in_channels=seq_len // 4,
                                  out_channels=seq_len // 2,
                                  kernel_size=3, padding=1, rounting_func=routing_function_22, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len // 2),
            nn.ReLU(),

            nn.ConvTranspose2d(seq_len // 2, seq_len // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(seq_len // 4),
            nn.ReLU(),

            AdaptiveRotatedConv2d(in_channels=seq_len // 4,
                                  out_channels=seq_len // 8,
                                  kernel_size=3, padding=1, rounting_func=routing_function_31, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len // 8),
            nn.ReLU(),
            AdaptiveRotatedConv2d(in_channels=seq_len // 8,
                                  out_channels=seq_len // 4,
                                  kernel_size=3, padding=1, rounting_func=routing_function_32, bias=True,
                                  kernel_number=4),
            nn.BatchNorm2d(seq_len // 4),
            nn.ReLU(),

            nn.ConvTranspose2d(seq_len // 4, seq_len // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(seq_len // 8),
            nn.ReLU(),

            nn.Conv2d(seq_len // 8, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        """

        :param x: [batch_size, seq_len, hidden_dim]
        :return:  [batch_size, 1, memory_width, memory_width]
        """

        batch_size = x.shape[0]
        x_l = torch.matmul(x.unsqueeze(-1), self.p_l.unsqueeze(0).expand(batch_size, -1, -1, -1))  # [batch_size, seq_len, hidden_dim, memory_width // 8]
        x_r = torch.matmul(x.unsqueeze(-1), self.p_r.unsqueeze(0).expand(batch_size, -1, -1, -1))  # [batch_size, seq_len, hidden_dim, memory_width // 8]

        x = torch.matmul(torch.transpose(x_l, -2, -1), x_r)  # [batch_size, seq_len, memory_width // 8, memory_width // 8]

        x = self.conv_layers(x)  # 卷积调整尺寸

        return x
