import torch
import torch.nn.functional as F


class SPMNWriteLoss(torch.nn.Module):
    def __init__(self):
        super(SPMNWriteLoss, self).__init__()

    def forward(self, M, M0, sentence_embedding):
        _M = M - M0

        loss = self.count_loss(_M, sentence_embedding)

        return loss

    def count_loss(self, input, sentence_embedding):
        bs = input.size(0)
        input_flat = input.view(bs, -1)
        input_flat = F.normalize(input_flat, p=2, dim=1)
        sentence_embedding_flat = sentence_embedding[:, 0]
        sentence_embedding_flat = F.normalize(sentence_embedding_flat, p=2, dim=1)

        similarities1 = torch.matmul(input_flat, input_flat.T)
        similarities2 = torch.matmul(sentence_embedding_flat, sentence_embedding_flat.T)

        # 计算相似度比值
        similarity_ratios = similarities1 / similarities2

        # 计算损失，目标是使得比值接近 1
        upper_indices = torch.triu_indices(bs, bs, offset=1)
        similarity_ratios = similarity_ratios[upper_indices[0], upper_indices[1]]

        loss = torch.mean((similarity_ratios - 1) ** 2 / 2)

        return loss


class AgentTrainLoss(torch.nn.Module):
    def __init__(self):
        super(AgentTrainLoss, self).__init__()

    def forward(self):


