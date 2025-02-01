import torch
from torch import nn

from spmn.mask import Mask


class Spmn(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, input_img_dim: int = 0, input_seq_len: int = 0):
        super(Spmn, self).__init__()

        self.mask_r = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=False)
        self.mask_u = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=True)
        self.mask_a = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=False)

        self.memory_width = memory_width
        self.memory_deep = memory_deep

        self.active = nn.ReLU()

    def forward(self, x, M):
        """

        :param x: 输入特征（batch_size, input_seq_len， input_hidden_dim）
        :param M: 记忆（memory_deep， memory_width， memory_width）
        :return:
        """

        # （batch_size, memory_deep， memory_width， memory_width）
        r = self.mask_r(x)   # write 0 ~ 1
        u = self.mask_u(x)  # write -1 ~ 1
        a = self.mask_a(x)  # read 0 ~ 1

        read = a * M

        _M = self.active(M * r)
        M = M + _M * u
        # M = self.normalize(M)
        return M, read

    def normalize(self, x):
        """

        :param x: (batch_size, memory_deep, memory_width, memory_width)
        :return: (batch_size, memory_deep, memory_width, memory_width)
        """
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.memory_deep, -1)

        # 计算每个向量的 L2 范数
        norm = torch.norm(x_reshaped, p=2, dim=-1, keepdim=True)

        # 归一化
        x_normalized = x_reshaped / norm  # 每个向量除以它的 L2 范数

        x_normalized = x_normalized.view(batch_size, self.memory_deep, self.memory_width, self.memory_width)

        return x_normalized
