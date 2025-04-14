from torch import nn
import torch

from spmn.backbone import efficientnet_b0, test_output


class MemoryEncoder(nn.Module):
    def __init__(self, c_out, output_dim, memory_width):
        super().__init__()

        self.c_out = c_out
        self.backbone = efficientnet_b0(c_out)
        self.head = nn.Linear(test_output(memory_width, self.backbone) ** 2, output_dim)

    def forward(self, x):
        """
        x: M [bs, deep, width, width]
        output: [memory_deep, bs, recall_num, input_dim]
        """
        bs = x.shape[0]
        deep = x.shape[1]

        # 0. ÐÞ¸ÄÎ¬¶È
        x = x.unsqueeze(2)  # [bs, deep, 1, width, width]
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])  # [bs * deep, 1, width, width]

        x = self.backbone(x)
        # print(x.shape)  # [bs * deep, recall_num, _, _]
        x = x.view(deep * bs, self.c_out, -1)  # [memory_deep, bs, recall_num, input_dim]

        return self.head(x)
