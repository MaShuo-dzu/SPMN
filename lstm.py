import torch
import torch.nn as nn
import time


# 定义一个大参数量的 LSTM 模型
class LargeLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=512, num_layers=1):
        super(LargeLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # Forward pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, hidden_size)
        return lstm_out


# 创建模型实例
model = LargeLSTM()


# 打印模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Model parameters: {count_parameters(model):,}")  # 确认模型参数量

# 测试运行时间
input_tensor = torch.randn(1, 256, 256)

start_time = time.time()
output = model(input_tensor)  # 前向传播
print(output.shape)
end_time = time.time()

print(f"Output shape: {output.shape}")
print(f"Time taken for forward pass: {end_time - start_time:.6f} seconds")
