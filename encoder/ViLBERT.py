# encoding=utf-8

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertConfig

from utils.module import CrossModalAttention


class ViLBERT(nn.Module):
    def __init__(self, backbone, text_encoder, hidden_dim, backbone_output_dim):
        """
        args:
            backbone: 图像特征提取模块，输出为[bs, backbone_output_dim, _, _] --> [bs, -1, backbone_output_dim]
            text_encoder: 文本特征提取模块，输出为[bs, seq_len, hidden_dim]
            forward: -->> [bs, seq_len, hidden_dim]
        """
        super().__init__()
        # 文本流：BERT
        self.text_encoder = text_encoder
        # 图像流：假设已提取图像特征 (e.g., Faster R-CNN)
        self.backbone = backbone
        self.backbone_output_dim = backbone_output_dim
        self.image_proj = nn.Linear(backbone_output_dim, hidden_dim)  # 将图像特征投影到文本空间

        # 跨模态注意力层
        self.cross_attn = CrossModalAttention(hidden_dim)

    def forward(self, text_input, image_input):
        bs = image_input.shape[0]
        # 编码文本
        text_output = self.text_encoder(**text_input).last_hidden_state

        # 编码图像
        image_feat = self.backbone(image_input)
        image_feat = image_feat.view(bs, -1, self.backbone_output_dim)
        image_proj = self.image_proj(image_feat)  # [batch, image_len, hidden_dim]

        # 跨模态交互
        fused_feat = self.cross_attn(text_output, image_proj)
        return fused_feat  # 融合后的特征


# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = models.efficientnet_b0(pretrained=False)
    # model = models.mobilenet_v2(pretrained=False)
    # 提取特征提取部分作为 backbone
    backbone = model.features
    text_encoder = BertModel(BertConfig())
    model = ViLBERT(
        backbone=backbone, text_encoder=text_encoder, hidden_dim=768, backbone_output_dim=1280
    )

    # 创建随机输入数据
    # 文本输入：模拟 BERT 的输入格式
    batch_size = 2
    text_len = 20

    # 文本输入
    text_input_ids = torch.randint(0, 1000, (batch_size, text_len))  # 随机词 ID
    text_attention_mask = torch.ones((batch_size, text_len))  # 注意力掩码
    text_input = {"input_ids": text_input_ids, "attention_mask": text_attention_mask}

    # 图像输入：假设图像特征维度为 2048
    image_feat = torch.randn(batch_size, 3, 224, 224)

    # 前向传播
    output = model(text_input, image_feat)
    print(output.shape)
