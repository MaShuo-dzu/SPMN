import torch
from torch import Tensor

from utils.data import NpzData, AgentTrainIter

each_iter = NpzData(0, torch.Tensor([0.1, 0.4, 0.1, 0.4]), torch.Tensor([2, 5, 7, 10]))

memory_number = len(each_iter.similarity)
linear_sequence = torch.linspace(0, 1, steps=memory_number)

c_pass = each_iter.similarity > 0.5
similarity: Tensor = each_iter.similarity[c_pass]  # [real_num]
print(similarity)

index: Tensor = each_iter.index[c_pass]  # [real_num]
print(index)
embeddings: Tensor = torch.ones(len(similarity), 5)

p: Tensor = linear_sequence[c_pass]  # [real_num]
print(p)

train_iter = AgentTrainIter(torch.ones(5), torch.cat((similarity.unsqueeze(-1), p.unsqueeze(-1), embeddings), dim=1))
print(train_iter.target)
print(train_iter.embedding.shape)
