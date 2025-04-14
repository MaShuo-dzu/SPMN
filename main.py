# encoding=utf-8
import time

import torch

from spmn.spmn import Spmn
from utils.tools import count_parameters

if __name__ == '__main__':
    # 设置超参数
    batch_size = 20
    seq_len = 10
    input_dim = 768
    memory_deep = 5
    memory_width = 224
    recall_num = 500
    output_dim = 768

    # 创建模型实例
    model = Spmn(
        memory_deep=memory_deep,
        memory_width=memory_width,
        recall_num=recall_num,
        input_dim=input_dim,
        output_dim=output_dim
    )

    print(count_parameters(model))

    # 创建输入张量
    x = torch.randn(batch_size, seq_len, input_dim)  # 输入特征

    # 创建记忆矩阵 M
    M = torch.randn(memory_deep, memory_width, memory_width)

    # 前向传播
    start_time = time.time()
    time_out, data_out, updated_M = model(x, M)
    end_time = time.time()
    print("cost time: ", end_time - start_time)

    # 验证输出形状
    assert time_out.shape == (batch_size, recall_num,
                              2), f"Time output shape mismatch. Expected: {(batch_size, recall_num, 2)}, Got: {time_out.shape}"
    assert data_out.shape == (batch_size, recall_num,
                              output_dim), f"Data output shape mismatch. Expected: {(batch_size, recall_num, output_dim)}, Got: {data_out.shape}"
    assert updated_M.shape == (memory_deep, memory_width,
                               memory_width), f"Memory shape mismatch. Expected: {(memory_deep, memory_width, memory_width)}, Got: {updated_M.shape}"

    print("Test passed! All output shapes are correct.")
