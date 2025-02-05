import torch


def time_distance(keep_memory, surplus_memory):
    """

    :param keep_memory: [1, output_dim]
    :param surplus_memory: [surplus, output_dim]
    :return: [bs, surplus, 1]
    """
    keep_p = keep_memory[0]  # [1]
    surplus_p = surplus_memory[:, 0]  # [surplus, 1]

    return torch.abs(keep_p - surplus_p)  # [surplus, 1]


def nms(output, nms_threshold: float = 0.000001):
    """
    Compared with a traditional nms, the nms_threshold is variable and decreases
    over time, usually 0.3 to 0.7 times the b threshold.

    :param output: [recall_num, output_dim]
    :param nms_threshold: Anything greater than this threshold will be suppressed
    :return: [search_num, output_dim]
    """

    conf_scores = output[:, 1]  # confs [recall_num]

    conf_scores, indices = conf_scores.sort(descending=True)
    output = output[indices]

    keep = []
    while output.numel() > 0:
        keep.append(output[0])
        dp = time_distance(output[0], output[1:])
        output = output[1:][dp >= nms_threshold]

    return torch.stack(keep, dim=0)
