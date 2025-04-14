# encoding=utf-8

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """跨模态注意力层（x2→x1）"""

    def __init__(self, dim):
        super().__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, 2 * dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        # text_feat: [batch, x1, dim]
        # image_feat: [batch, x2, dim]
        Q = self.q_linear(x1)
        K, V = self.kv_linear(x2).chunk(2, dim=-1)
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, V)
