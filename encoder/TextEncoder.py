# encoding=utf-8

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class TextAttentionModel(nn.Module):
    def __init__(self, text_encoder, hidden_dim):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, text_input):
        # text_input 是字典：{'input_ids': ..., 'attention_mask': ...}
        text_output = self.text_encoder(**text_input).last_hidden_state  # [bs, seq_len, hidden_dim]

        return text_output


# 测试代码
if __name__ == "__main__":
    # 构造模型
    text_encoder = BertModel(BertConfig())
    model = TextAttentionModel(text_encoder=text_encoder, hidden_dim=768)

    # 构造随机输入
    batch_size = 2
    text_len = 12
    text_input_ids = torch.randint(0, 1000, (batch_size, text_len))  # 模拟输入
    text_attention_mask = torch.ones((batch_size, text_len))

    text_input = {"input_ids": text_input_ids, "attention_mask": text_attention_mask}

    # 前向传播
    output = model(text_input)
    print(output.shape)  # 预期输出：[batch_size, hidden_dim]
