from torch import nn
import torch.nn.functional as F

from spmn.mask import Mask
from spmn.recall import Recall


class Spmn(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, input_img_dim: int = 0, input_seq_len: int = 0, recall_num: int = 500, output_dim: int = 2048):
        super(Spmn, self).__init__()

        self.recall_num = recall_num
        self.output_dim = output_dim

        self.mask_r = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=False)
        self.mask_u = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=True)
        self.mask_a = Mask(seq_len=input_seq_len, img_dim=input_img_dim, memory_width=memory_width, memory_deep=memory_deep, active_tanh=True)

        self.memory_width = memory_width
        self.memory_deep = memory_deep

        self.recall_block = Recall(memory_width, memory_deep, recall_num, output_dim)

    def forward(self, x, M):
        """

        :param x: 输入特征（batch_size, input_seq_len， input_hidden_dim）
        :param M: 记忆（memory_deep， memory_width， memory_width）
        :return:
        """

        # （batch_size, memory_deep， memory_width， memory_width）
        # 更新门计算
        r = self.mask_r(x)   # 0 ~ 1

        # 重置门计算
        a = self.mask_a(x)  # -1 ~ 1

        # 候选隐藏状态
        u = self.mask_u(x)  # -1 ~ 1
        _M = F.tanh(M * a * u)

        # 最终隐藏状态
        M = (1 - r) * M + _M * r

        # 计算output
        # （batch_size, memory_deep， memory_width， memory_width） -> （batch_size， recall_num， 2 + output_dim）
        output = self.recall_block(_M)

        return M, output
