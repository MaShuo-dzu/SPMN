import torch
import torch.nn as nn

from spmn.compress import Compress
from spmn.encoder import TextEncoder, ImgEncoder


class Mask(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, input_dim: int):
        super(Mask, self).__init__()

        assert memory_width / 8 == int(memory_width / 8), "memory_width 必须是8的倍数 ！"

        self.encoder = TextEncoder(input_dim, memory_width)

        self.active = nn.Tanh()

        self.compress_list = nn.ModuleList()
        for i in range(2, memory_deep + 1):
            self.compress_list.append(Compress(2))

    def forward(self,  x):
        """

        :param x: shape (batch_size, input_dim)
        :return o: shape (batch_size, memory_deep, memory_width, memory_width)
        """

        o_list = []
        encoder_x = self.encoder(x)  # (batch_size, 1, memory_width, memory_width)
        _x = encoder_x.clone()
        o_list.append(_x)

        for compress in self.compress_list:
            _x = compress(torch.cat([_x, encoder_x], dim=1))
            _x = self.active(_x)
            o_list.append(_x)

        o = torch.cat(o_list, dim=1)
        return o
