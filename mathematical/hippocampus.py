import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class HippocampusLayer(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, C_m=1.0, E_rest=-65.0, V_rev=0.0, g_max=1.0, eta=0.01, delta_t=0.1):
        super(HippocampusLayer, self).__init__()

        # 初始化神经网络层的参数
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.C_m = C_m
        self.E_rest = E_rest
        self.V_rev = V_rev
        self.g_max = g_max
        self.eta = eta
        self.delta_t = delta_t

        # 将膜电位从 nn.Parameter 改为普通 Tensor（因为它是动态更新的）
        self.V = torch.full((num_neurons,), E_rest)  # 每个神经元的膜电位

        # 初始化突触电导（g_ij），随机初始化
        self.g = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)

        # 外部输入电流
        self.I_ext = nn.Parameter(torch.zeros(num_neurons))

        # 活动函数（突触前和突触后神经元的活动）
        self.pre_spike_times = torch.zeros(num_neurons)  # 突触前神经元的发放时刻
        self.post_spike_times = torch.zeros(num_neurons)  # 突触后神经元的发放时刻

    def forward(self, spikes):
        # 计算突触电流
        I_syn = torch.matmul(self.g, self.V - self.V_rev)

        # 计算膜电位的变化
        dV = (-self.V + self.E_rest + I_syn + self.I_ext) / self.tau_m
        self.V = self.V + self.delta_t * dV  # 更新膜电位

        # 计算神经元活动（是否发放）
        spiking = self.V >= 0  # 当膜电位超过阈值，神经元发放
        self.V[spiking] = self.E_rest  # 重置膜电位为静息电位

        # 更新突触可塑性（LTP/LTD）
        self.update_synaptic_plasticity(spikes)

        return spiking.float()

    def update_synaptic_plasticity(self, spikes):
        # 更新突触前和突触后活动
        for i in range(self.num_neurons):
            A_pre = self.pre_spike_times[i]  # 突触前神经元活动
            A_post = self.post_spike_times[i]  # 突触后神经元活动

            # 计算突触电导的变化
            delta_g = self.eta * (A_pre * A_post - self.g[i, i] / self.g_max)

            # 克隆 `self.g[i, i]` 并更新
            self.g.data[i, i] = self.g.data[i, i] + delta_g  # 使用 .data 来避免视图问题

        # 在这里，A_pre和A_post的计算可以是基于时间窗的脉冲到达（或时间衰减函数）


# 示例使用
num_neurons = 5
layer = HippocampusLayer(num_neurons)

# 模拟输入脉冲
spikes = torch.randint(0, 2, (num_neurons,))  # 随机生成神经元是否发放的脉冲

# 前向传播
start_time = time.time()
print(spikes.shape)
spiking = layer(spikes)
end_time = time.time()
print(spiking, spiking.shape)
print("cost time: ", end_time - start_time)

