import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class FourierSeriesLayer(nn.Module):
    """自定义层实现傅里叶级数的功能"""

    def __init__(self, num_harmonics=10):
        super(FourierSeriesLayer, self).__init__()
        self.num_harmonics = num_harmonics

        # 可学习的振幅系数
        self.amplitudes_sin = nn.Parameter(torch.randn(num_harmonics))
        self.amplitudes_cos = nn.Parameter(torch.randn(num_harmonics))

        # 可学习的频率系数 (从1开始的整数倍基频)
        self.frequencies = nn.Parameter(torch.arange(1, num_harmonics + 1, dtype=torch.float32), requires_grad=False)

        # 可学习的相位偏移
        self.phases_sin = nn.Parameter(torch.randn(num_harmonics))
        self.phases_cos = nn.Parameter(torch.randn(num_harmonics))

        # 可学习的直流分量
        self.dc_offset = nn.Parameter(torch.randn(1))

    def forward(self, x):
        """
        前向传播计算傅里叶级数:
        f(x) = DC + Σ [a_n*sin(nωx + φ_n) + b_n*cos(nωx + θ_n)]
        """
        batch_size = x.size(0)
        x_expanded = x.view(batch_size, 1)  # [batch_size, 1]

        # 计算所有谐波的sin和cos部分
        sin_part = torch.zeros(batch_size, self.num_harmonics)
        cos_part = torch.zeros(batch_size, self.num_harmonics)

        for i in range(self.num_harmonics):
            # 计算第i个谐波的角度: n*ω*x + φ
            angle_sin = self.frequencies[i] * x_expanded + self.phases_sin[i]
            angle_cos = self.frequencies[i] * x_expanded + self.phases_cos[i]

            # 计算sin和cos部分，并乘以相应的振幅
            sin_part[:, i] = self.amplitudes_sin[i] * torch.sin(angle_sin).squeeze()
            cos_part[:, i] = self.amplitudes_cos[i] * torch.cos(angle_cos).squeeze()

        # 对所有谐波求和，并加上直流分量
        output = torch.sum(sin_part + cos_part, dim=1, keepdim=True) + self.dc_offset
        return output


class FourierSeriesNetwork(nn.Module):
    """完整的傅里叶级数神经网络模型"""

    def __init__(self, num_harmonics=10):
        super(FourierSeriesNetwork, self).__init__()
        self.fourier_layer = FourierSeriesLayer(num_harmonics)

    def forward(self, x):
        return self.fourier_layer(x)


def generate_data(func, x_range=(-np.pi, np.pi), num_samples=1000):
    """生成训练数据"""
    x = np.linspace(x_range[0], x_range[1], num_samples)
    y = func(x)

    # 转换为PyTorch张量
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return TensorDataset(x_tensor, y_tensor)


def train_model(model, dataloader, epochs=1000, lr=0.01):
    """训练傅里叶级数神经网络"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.6f}')

    return model


def visualize_results(model, func, x_range=(-np.pi, np.pi), num_samples=500):
    """可视化模型拟合效果"""
    x = np.linspace(x_range[0], x_range[1], num_samples)
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)

    # 计算真实函数值和模型预测值
    y_true = func(x)
    with torch.no_grad():
        y_pred = model(x_tensor).numpy().flatten()

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'b-', label='True Function')
    plt.plot(x, y_pred, 'r--', label='Fourier Series Approximation')
    plt.legend()
    plt.title('Fourier Series Neural Network Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


# 示例：训练模型拟合方波函数
def square_wave(x):
    """方波函数"""
    return np.where(np.sin(x) > 0, 1, -1)


# 创建模型和数据
model = FourierSeriesNetwork(num_harmonics=20)
dataset = generate_data(square_wave)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
trained_model = train_model(model, dataloader, epochs=1000, lr=0.01)

# 可视化结果
visualize_results(trained_model, square_wave)
