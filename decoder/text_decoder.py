import torch
from torch import nn


class MobileLLMDecoder(nn.Module):
    def __init__(self):
        super(MobileLLMDecoder, self).__init__()


class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(LstmDecoder, self).__init__()
        # ����LSTM��
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        # ��������㣬��LSTM�����ӳ�䵽����ռ�
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, x, hidden):
        # x: �������У���״Ϊ(batch_size, seq_len, input_size)
        output, hidden = self.lstm(x, hidden)
        # ��LSTM�����ͨ��ȫ���Ӳ�
        output = self.fc(output[:, -1, :])  # ֻȡ���е����һ��ʱ�䲽�����
        return output, hidden

    def init_hidden(self, batch_size):
        # ��ʼ��LSTM������״̬
        num_directions = 1  # ����LSTM
        return (torch.zeros(self.num_layers * num_directions, batch_size, self.lstm.hidden_size),
                torch.zeros(self.num_layers * num_directions, batch_size, self.lstm.hidden_size))
