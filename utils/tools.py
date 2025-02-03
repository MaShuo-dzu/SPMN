import os

import torch
from torch import nn


import json


def save_arg(data_dict: dict, path: str):
    filename = 'arg.json'

    # 将字典保存到JSON文件
    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    print(f'arg has been written to {os.path.join(path, filename)}')


def find_longest_sentence(file_path):
    max_length = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空白字符
            if line:  # 确保行不为空
                length = len(line)
                if length > max_length:
                    max_length = length
    return max_length


def count_parameters(model: nn.Module):
    """
    统计 PyTorch 模型的参数数量
    :param model: PyTorch 模型
    :return: 参数总数和可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())  # 参数总数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数
    return total_params, trainable_params


def calculate_stats(tensor):
    """
    计算并返回给定张量的最大值、最小值、平均值和方差。

    参数:
    tensor (torch.Tensor): 输入的PyTorch张量。

    返回:
    tuple: 包含最大值、最小值、平均值和方差的元组。
    """
    # 计算最大值
    max_value = torch.max(tensor)

    # 计算最小值
    min_value = torch.min(tensor)

    # 计算平均值
    mean_value = torch.mean(tensor)

    # 计算方差
    variance = torch.var(tensor)

    return max_value, min_value, mean_value, variance
