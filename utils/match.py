import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_matching(pred_points, true_points):
    """
    使用匈牙利算法匹配预测点和实际点。

    Args:
        pred_points (numpy.ndarray): 预测点，形状为 (batch_size, pred_nums, 2)。
                                      每个点包含位置和置信度。
                                      位置范围 [0, 1], 置信度范围 [0, 1]。
        true_points (numpy.ndarray): 实际点，形状为 (batch_size, true_nums, 2)。
                                      每个点包含位置和置信度。

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: 每个批次的匹配结果，形式为 (indices_pred, indices_true)。
                                              如果预测点数量少于实际点，则匹配所有预测点。
                                              如果预测点数量多于实际点，则匹配所有实际点。
    """
    batch_size, pred_nums, _ = pred_points.shape
    _, true_nums, _ = true_points.shape

    result_list = []

    for batch_idx in range(batch_size):
        # 提取当前批次的预测点和真实点
        pred_batch = pred_points[batch_idx]
        true_batch = true_points[batch_idx]

        # 提取位置和置信度
        pred_pos = pred_batch[:, 0:1]  # 取位置部分
        pred_conf = pred_batch[:, 1:]  # 取置信度部分
        true_pos = true_batch[:, 0:1]  # 取位置部分
        # true_conf = true_batch[:, 1:]  # 取置信度部分

        # 计算成本矩阵
        # 使用欧氏距离作为位置成本，并结合置信度
        cost_matrix = np.linalg.norm(pred_pos[:, None] - true_pos[None, :], axis=2)  # (pred_nums, true_nums)
        cost_matrix *= 1 / (pred_conf + 1e-8)  # 加权置信度，低置信度的点会被优先匹配

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 限制匹配数目为较小的数目
        if pred_nums < true_nums:
            result = (row_ind, col_ind[:pred_nums])
        else:
            result = (row_ind[:true_nums], col_ind)

        result_list.append(result)

    return result_list


# 示例用法
if __name__ == "__main__":
    np.random.seed(42)
    # 随机生成预测点和真实点
    batch_size = 2
    pred_nums = 5
    true_nums = 3

    pred_points = np.random.rand(batch_size, pred_nums, 2)
    true_points = np.random.rand(batch_size, true_nums, 2)

    # 调用匹配函数
    matches = hungarian_matching(pred_points, true_points)

    # 输出结果
    for batch_idx, (pred_indices, true_indices) in enumerate(matches):
        print(f"Batch {batch_idx}:")
        print(f"  Predicted indices: {pred_indices}")
        print(f"  True indices:      {true_indices}")
