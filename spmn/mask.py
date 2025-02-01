import torch
import torch.nn as nn

from spmn.compress import Compress
from spmn.encoder import TextEncoder, ImgEncoder


class Mask(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, seq_len: int = 0, img_dim: int = 0, active_tanh: bool = True):
        super(Mask, self).__init__()

        assert img_dim * seq_len == 0 and seq_len + img_dim > 0, "seq_len和img_dim有且仅需要有1个。"

        assert memory_width / 8 == int(memory_width / 8), "memory_width 必须是8的倍数 ！"

        self.active_tanh = active_tanh

        self.encoder = ImgEncoder(img_dim, memory_width) if img_dim != 0 else TextEncoder(seq_len, memory_width)

        self.active = nn.Tanh() if active_tanh else nn.Sigmoid()

        self.compress_list = nn.ModuleList()
        for i in range(2, memory_deep + 1):
            self.compress_list.append(Compress(2))

    def forward(self,  x):
        """

        :param x: shape (batch_size, seq_len, hidden_dim)
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
