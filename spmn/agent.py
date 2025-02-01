import torch.nn as nn
from transformers import AutoModel

from decoder.text_decoder import LstmDecoder
from spmn.spmn import Spmn


class BgeLstmAgent(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int,
                 seq_len: int,
                 lstm_seq_len: int,
                 lstm_hidden_size: int,
                 lstm_layers: int,
                 encoder: str,
                 vocab_size: int):
        super(BgeLstmAgent, self).__init__()

        self.memory_width = memory_width
        self.memory_deep = memory_deep
        self.vocab_size = vocab_size

        self.Encoder = AutoModel.from_pretrained(encoder)
        self.Spmn = Spmn(memory_width, memory_deep)
        self.Decoder = LstmDecoder(lstm_seq_len, lstm_hidden_size, vocab_size, lstm_layers)

    def encoder(self, x, M):
        """
        �������
        :param M: ��������
        :param x: [bs, seq_len]  ������ı�����
        :return: x
        """
        x = self.Encoder(x)  # [bs, seq_len, hidden_size]
        M, read = self.Spmn(x, M)
        return x, M, read

    def zip_input(self, x_old, x_new):
        """

        :param x_old:  �Ѿ����ڣ����ɣ��ľ�������
        :param x_new:  decoder�����ɵĴ�����
        :return:  [bs, lstm_seq_len, hidden_size]
        """

    def decoder(self, x, read):
        """
        ��Ӧ
        :param x: �Իع����� [bs, lstm_seq_len, hidden_size]
        :param read: ��������  [bs, memory_deep, memory_width, memory_width]
        :return:
        """
