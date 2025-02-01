import torch

input = torch.tensor([
        [2, 3],
        [1, 7],
        [4, 5]
])
print(input.shape)

w = torch.tensor([
        [2, 3],
        [1, 2]
])

o = input @ w
o_T = input @ w.T
print(o)
print(o_T)
