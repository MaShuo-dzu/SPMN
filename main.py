import torch

from spmn.spmn import Spmn
from utils.tools import calculate_stats, count_parameters

import time

# print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建记忆张量 （10, 256, 256）
memory_deep = 10
memory_width = 512
M = torch.zeros(memory_deep, memory_width, memory_width).to(device)
print(calculate_stats(M))

# 模拟输入 （bs， 1024）
batch_size = 1
input_dim = 2048
input = torch.randn(batch_size, input_dim).to(device)

model = Spmn(memory_width=memory_width, memory_deep=memory_deep, input_dim=input_dim
             ).to(device)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

print("模型参数量/训练参数量： ", count_parameters(model))

start_time = time.time()
# 前向传播
output = model(input)
M = model.get_M()
print(f"M shape: {M.shape}")
print(f"o shape: {output.shape}")
print(calculate_stats(M))

end_time = time.time()
print("cost time: ", end_time - start_time)
