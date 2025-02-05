import torch
import torch.nn.functional as F

from utils.match import hungarian_matching


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
    def __init__(self, p_rate: float = 0.7, c_rate: float = 0.3, d_rate: float = 1):
        super(AgentTrainLoss, self).__init__()

        self.p_rate = p_rate
        self.c_rate = c_rate
        self.d_rate = d_rate

    def forward(self, output: list, target):
        """

        :param output: [[search_num, output_dim]...]  len = bs 根据conf阈值筛选之后的输出
        :param target: [[real_num, output_dim]...] len = bs
        :return:
        """
        bs = len(output)

        loss_list = []
        for each_bs in range(bs):
            o = output[each_bs]  # [search_num, output_dim]
            t = target[each_bs]  # [real_num, output_dim]

            # 匹配
            pred_indices, true_indices = hungarian_matching(o, t)
            pred = o[pred_indices]
            true = t[true_indices]

            # 时间损失 (p)
            loss_t = F.smooth_l1_loss(pred[:, 0], true[:, 0], reduction='mean', beta=1.0)
            # 置信度损失
            loss_c = F.smooth_l1_loss(pred[:, 1], true[:, 1], reduction='mean', beta=1.0)
            # 内容损失 (data + conf)
            loss_data = 1 - F.cosine_similarity(pred[:, 2:], true[:, 2:], dim=1)  # dim=1 表示沿着向量的维度计算
            loss_data = torch.sum(loss_data)

            loss_list.append(loss_t.item() * self.p_rate + loss_c.item() * self.c_rate + loss_data.item() * self.d_rate)

        loss = sum(loss_list) / len(loss_list)

        return loss

