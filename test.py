import torch

from spmn.recall import Recall

memory_width = 64
memory_deep = 16
recall_num = 500
output_dim = 2048

model = Recall(memory_width=memory_width, memory_deep=memory_deep, recall_num=recall_num, output_dim=output_dim)

batch_size = 2
x = torch.randn(batch_size, memory_deep, memory_width, memory_width)   # (batch_size, in_channel, memory_width, memory_width)

output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")