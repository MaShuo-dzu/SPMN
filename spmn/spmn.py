import torch
from torch import nn
import torch.nn.functional as F

from spmn.mask import Mask
from spmn.recall import Recall


class Spmn(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, input_dim: int = 2048, recall_num: int = 500, output_dim: int = 384):
        super(Spmn, self).__init__()

        self.recall_num = recall_num
        self.output_dim = output_dim

        self.mask_r = Mask(input_dim=input_dim, memory_width=memory_width, memory_deep=memory_deep)
        self.mask_u = Mask(input_dim=input_dim, memory_width=memory_width, memory_deep=memory_deep)
        self.mask_a = Mask(input_dim=input_dim, memory_width=memory_width, memory_deep=memory_deep)

        self.memory_width = memory_width
        self.memory_deep = memory_deep

        self.recall_block = Recall(memory_width, memory_deep, recall_num, output_dim)

        self.M = None

    def init_M(self, batch_num):
        self.M = torch.zeros(batch_num, self.memory_deep,self. memory_width, self.memory_width)

    def get_M(self):
        return self.M

    def forward(self, x):
        """

        :param x: 输入特征（batch_size, input_dim）
        :return:
        """
        assert self.M is not None, "请在运行模型之前初始化M"

        M = self.M.clone().to(x.device).detach()

        # （batch_size, memory_deep， memory_width， memory_width）
        # 更新门计算
        r = self.mask_r(x)

        # 重置门计算
        a = self.mask_a(x)

        # 候选隐藏状态
        u = self.mask_u(x)
        _M = F.relu(M * a * u)

        # 最终隐藏状态
        M = (1 - r) * M + _M * r
        self.M = M.clone()

        # 计算output
        # （batch_size, memory_deep， memory_width， memory_width） -> （batch_size， recall_num， 2 + output_dim）
        output = self.recall_block(_M)

        return output

    def version(self):
        return "0.0.1"
