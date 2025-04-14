import torch
from torch import nn
import torch.nn.functional as F

from spmn.MemoryEncoder import MemoryEncoder
from utils.module import CrossModalAttention


class Spmn(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, input_dim: int = 2048, recall_num: int = 500, output_dim: int = 384):
        super(Spmn, self).__init__()

        self.recall_num = recall_num
        self.output_dim = output_dim

        self.memory_width = memory_width
        self.memory_deep = memory_deep

        self.MemoryEncoder = MemoryEncoder(
            c_out=recall_num,
            output_dim=input_dim,
            memory_width=memory_width
        )

        self.CrossAtnLayers = nn.ModuleList()
        self.CrossAtnFeedForward = nn.ModuleList()
        self.CrossAtnNorm = nn.ModuleList()
        self.GateLeftLayer = nn.ModuleList()
        self.GateRightLayer = nn.ModuleList()

        for _ in range(memory_deep):
            self.CrossAtnLayers.append(CrossModalAttention(input_dim))
            self.CrossAtnFeedForward.append(
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(input_dim // 2, input_dim),
                )
            )
            self.CrossAtnNorm.append(nn.LayerNorm(input_dim))
            self.GateLeftLayer.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.memory_width),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
            )
            self.GateRightLayer.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.memory_width),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
            )

        self.zGate = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.rGate = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.hGate = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Tanh()
        )

        self.TimeHead = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(input_dim // 2, 2),
                )
        self.DataHead = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(input_dim // 2, output_dim),
                )

    def forward(self, x, M=None):
        """

        :param x: 输入特征（batch_size, seq_len, input_dim）
        :param M: 输入特征 [memory_deep, memory_width, memory_width]
        :return: [bs, recall_num, 2], [bs, recall_num, output_dim], [memory_deep, memory_width, memory_width]
        """
        bs = x.shape[0]

        if M is None:
            M = torch.zeros(size=[self.memory_deep, self.memory_width, self.memory_width], dtype=torch.float, device=x.device)

        # M encoder
        M_copy = M.clone().repeat(bs, 1, 1, 1).to(x.device)  # [bs, memory_deep, memory_width, memory_width]
        m_feature = self.MemoryEncoder(M_copy)  # [memory_deep, bs, recall_num, input_dim]

        # 交叉注意力采样 (事件特征->M)
        m_layers_atn_feature_list = []  # list ([batch_size, recall_num, input_dim])
        z_s = []
        r_s = []
        for step in range(self.memory_deep):
            if step == 0:
                m_atn_feature = m_feature[step]
            else:
                m_atn_feature = m_feature[step] + m_layers_atn_feature_list[step - 1]

            m_atn_feature_o = self.CrossAtnLayers[step](m_atn_feature, x)  # [batch_size, recall_num, input_dim]
            m_atn_feature = self.CrossAtnNorm[step](m_atn_feature + m_atn_feature_o)
            m_atn_feature_linear = self.CrossAtnFeedForward[step](m_atn_feature)
            m_atn_feature = self.CrossAtnNorm[step](m_atn_feature + m_atn_feature_linear)
            m_layers_atn_feature_list.append(m_atn_feature)

            m_layer_gate = torch.mean(m_atn_feature, dim=0)  # [recall_num, input_dim]
            m_layer_left = self.GateLeftLayer[step](m_layer_gate)  # [recall_num, memory_width]
            m_layer_right = self.GateRightLayer[step](m_layer_gate)  # [recall_num, memory_width]
            # 更新门
            z = torch.mm(torch.transpose(m_layer_left, -2, -1), m_layer_right)  # [memory_width, memory_width]
            z_s.append(z)
            # 重置门
            r = torch.mm(torch.transpose(m_layer_left, -2, -1), m_layer_right)  # [memory_width, memory_width]
            r_s.append(r)

        # 记忆与遗忘 (GRU)
        z_M = torch.stack(z_s, dim=0)  # [memory_deep, memory_width, memory_width]
        r_M = torch.stack(r_s, dim=0)  # [memory_deep, memory_width, memory_width]
        z_M = self.zGate(z_M.unsqueeze(1)).squeeze(1)
        r_M = self.rGate(r_M.unsqueeze(1)).squeeze(1)
        # 计算候选隐藏状态
        _M = self.hGate((M * r_M).unsqueeze(1)).squeeze(1)
        # 隐藏状态更新
        M = z_M * M + (1 - z_M) * _M
        M.detach_()

        # 生成时间、内容
        time_out = self.TimeHead(m_layers_atn_feature_list[-1])
        data_out = self.DataHead(m_layers_atn_feature_list[-1])

        return time_out, data_out, M

    def version(self):
        return "0.0.2"
