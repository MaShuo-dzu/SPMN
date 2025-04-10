import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from decoder.text_decoder import MobileLLMDecoder
from spmn.spmn import Spmn


class BgeGPT3Agent(nn.Module):
    def __init__(self, memory_width: int, memory_deep: int, M: Tensor):
        super(BgeGPT3Agent, self).__init__()

        self.memory_width = memory_width
        self.memory_deep = memory_deep
        self.M = M

        self.Encoder = AutoModel.from_pretrained(r"G:\ms\SPMN\bge-small-zh")
        self.Spmn = Spmn(memory_width, memory_deep)
        self.Decoder = MobileLLMDecoder()

    def forward(self, x):
        x = self.Encoder(x)
        self.M, read = self.Spmn(x, self.M)
        x = self.Decoder(x, read)
        return x
