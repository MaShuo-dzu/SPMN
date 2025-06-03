import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义参数
tau = 10.0  # 时间常数
r = 0.5  # 不应期系数
w_EE = 0.1  # 兴奋性神经元到兴奋性神经元的连接强度
w_EI = 0.2  # 抑制性神经元到兴奋性神经元的连接强度
w_IE = 0.3  # 兴奋性神经元到抑制性神经元的连接强度
w_II = 0.4  # 抑制性神经元到抑制性神经元的连接强度
theta_E = 0.5  # 兴奋性神经元的阈值
theta_I = 0.5  # 抑制性神经元的阈值
a_E = 1.0  # 兴奋性神经元的增益参数
a_I = 1.0  # 抑制性神经元的增益参数

# 定义非线性激活函数（Sigmoid函数）
def f_E(u):
    return 1 / (1 + torch.exp(-a_E * (u - theta_E)))

def f_I(u):
    return 1 / (1 + torch.exp(-a_I * (u - theta_I)))

# 定义 Wilson-Cowan 群体层
class WilsonCowanLayer(nn.Module):
    def __init__(self):
        super(WilsonCowanLayer, self).__init__()
        self.w_EE = nn.Parameter(torch.tensor(w_EE))
        self.w_EI = nn.Parameter(torch.tensor(w_EI))
        self.w_IE = nn.Parameter(torch.tensor(w_IE))
        self.w_II = nn.Parameter(torch.tensor(w_II))
        self.r = nn.Parameter(torch.tensor(r))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.theta_E = nn.Parameter(torch.tensor(theta_E))
        self.theta_I = nn.Parameter(torch.tensor(theta_I))
        self.a_E = nn.Parameter(torch.tensor(a_E))
        self.a_I = nn.Parameter(torch.tensor(a_I))

    def forward(self, E_prev, I_prev, h_E, h_I):
        # 计算兴奋性神经元的输入
        input_E = self.w_EE * E_prev - self.w_EI * I_prev + h_E
        E_next = E_prev + (-E_prev + (1 - self.r * E_prev) * f_E(input_E)) / self.tau

        # 计算抑制性神经元的输入
        input_I = self.w_IE * E_prev - self.w_II * I_prev + h_I
        I_next = I_prev + (-I_prev + (1 - self.r * I_prev) * f_I(input_I)) / self.tau

        return E_next, I_next


# 构建模型
class WilsonCowanModel(nn.Module):
    def __init__(self):
        super(WilsonCowanModel, self).__init__()
        self.wc_layer = WilsonCowanLayer()

    def forward(self, h_E, h_I, num_steps=100):
        E = torch.zeros_like(h_E)
        I = torch.zeros_like(h_I)
        E_history = []
        I_history = []

        for _ in range(num_steps):
            E, I = self.wc_layer(E, I, h_E, h_I)
            E_history.append(E)
            I_history.append(I)

        return torch.stack(E_history), torch.stack(I_history)

# 定义外部输入函数
def h_E(t):
    return 0.1 * torch.sin(t)

def h_I(t):
    return 0.1 * torch.cos(t)

# 生成时间序列数据
time_steps = 1000
t_values = torch.linspace(0, 100, time_steps)
h_E_values = h_E(t_values)
h_I_values = h_I(t_values)

# 创建模型
model = WilsonCowanModel()

# 模拟动态过程
E_history = []
I_history = []

with torch.no_grad():
    for t in range(time_steps):
        h_E = h_E_values[t]
        h_I = h_I_values[t]
        E, I = model(h_E, h_I, num_steps=1)
        E_history.append(E.numpy())
        I_history.append(I.numpy())

# 可视化结果
plt.figure()
plt.plot(t_values.numpy(), E_history, label='Excitatory Neuron Activity')
plt.plot(t_values.numpy(), I_history, label='Inhibitory Neuron Activity')
plt.xlabel('Time')
plt.ylabel('Activity')
plt.legend()
plt.show()
