import torch

A = torch.randn(100, 10)

U, S, V = torch.svd(A)

k = torch.topk(S, k=min(5, S.numel()), largest=True, sorted=True)[1].numel()

U_k = U[:, :k]
S_k = S[:k]
V_k = V[:, :k]

A_reduced = torch.mm(U_k, torch.diag(S_k))

print(A_reduced.shape)