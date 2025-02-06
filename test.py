import torch

from utils.loss import AgentTrainLoss

loss_fn = AgentTrainLoss(p_rate=0.7, c_rate=0.3, d_rate=1.0)

batch_size = 2
output_dim = 5
max_search_num = 10
max_real_num = 000

output = [
    torch.rand(torch.randint(1, max_search_num, (1,)).item(), output_dim)
    for _ in range(batch_size)
]
target = [
    torch.rand(torch.randint(1, max_real_num, (1,)).item(), output_dim)
    for _ in range(batch_size)
] if max_real_num else [torch.Tensor([]), torch.Tensor([])]

loss = loss_fn(output, target)
print(f"Loss: {loss}")