import torch

from spmn.spmn import Spmn
from tools import calculate_stats, count_parameters

import time

# print(torch.__version__)

# 创建记忆张量 （10, 256, 256）
memory_deep = 10
memory_width = 512
M = torch.zeros(memory_deep, memory_width, memory_width)
print(calculate_stats(M))

# 模拟输入 （512， 1024）
seq_len = 77
hidden_dim = 512
input = torch.randn(seq_len, hidden_dim)

model = Spmn(memory_width=memory_width, memory_deep=memory_deep,
             input_seq_len=seq_len,
             # input_img_dim=512
             )
print("模型参数量/训练参数量： ", count_parameters(model))

start_time = time.time()
# 前向传播
output, _ = model(input, M)
print(f"M shape: {output.shape}")
print(calculate_stats(output))

end_time = time.time()
print("cost time: ", end_time - start_time)
