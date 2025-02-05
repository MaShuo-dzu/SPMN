import torch
from scipy.optimize import linear_sum_assignment


def hungarian_matching(pred_points, true_points):
    """
    使用匈牙利算法匹配预测点和实际点。

    Args:
        pred_points (torch.Tensor): 预测点，形状为 (pred_nums, output_dim)。
                                      每个点包含位置和置信度。
                                      位置范围 [0, 1], 置信度范围 [0, 1]。
        true_points (torch.Tensor): 实际点，形状为 (real_num, output_dim)。
                                      每个点包含位置和置信度。

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 每个批次的匹配结果，形式为 (indices_pred, indices_true)。
                                              如果预测点数量少于实际点，则匹配所有预测点。
                                              如果预测点数量多于实际点，则匹配所有实际点。
    """
    pred_nums, _ = pred_points.shape
    real_nums, _ = true_points.shape

    # 提取位置和置信度
    pred_pos = pred_points[:, 0:1]  # 取位置部分
    true_pos = true_points[:, 0:1]  # 取位置部分

    # 计算成本矩阵
    # 使用欧氏距离作为位置成本
    cost_matrix = torch.cdist(pred_pos, true_pos)  # (pred_nums, true_nums)

    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 限制匹配数目为较小的数目
    if pred_nums < real_nums:
        result = (row_ind, col_ind[:pred_nums])
    else:
        result = (row_ind[:real_nums], col_ind)

    return result


# 示例用法
# if __name__ == "__main__":
#     # 随机生成预测点和真实点
#     pred_nums = 3
#     true_nums = 5
#
#     pred_points = torch.randn(pred_nums, 2)
#     true_points = torch.randn(true_nums, 2)
#     print(pred_points)
#     print(true_points)
#
#     # 调用匹配函数
#     matches = hungarian_matching(pred_points, true_points)
#
#     # 输出结果
#     pred_indices, true_indices = matches
#     print(f"  Predicted indices: {pred_indices}")
#     print(f"  True indices:      {true_indices}")
