import torch
from torch import nn


class MobileLLMDecoder(nn.Module):
    def __init__(self):
        super(MobileLLMDecoder, self).__init__()


class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LstmDecoder, self).__init__()
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # 定义输出层，将LSTM的输出映射到输出空间
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, x, hidden):
        # x: 输入序列，形状为(batch_size, seq_len, input_size)
        output, hidden = self.lstm(x, hidden)
        # 将LSTM的输出通过全连接层
        output = self.fc(output[:, -1, :])  # 只取序列的最后一个时间步的输出
        return output, hidden

    def init_hidden(self, batch_size):
        # 初始化LSTM的隐藏状态
        num_directions = 1  # 单向LSTM
        return (torch.zeros(self.num_layers * num_directions, batch_size, self.lstm.hidden_size),
                torch.zeros(self.num_layers * num_directions, batch_size, self.lstm.hidden_size))
