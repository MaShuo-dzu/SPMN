# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from utils.tools import count_model_params, count_trainable_params

from spikingjelly.activation_based import neuron, layer, surrogate, encoding, functional


def autocorr_coefficient(x: torch.Tensor, lag: int = 1) -> torch.Tensor:
    """
    Compute the autocorrelation coefficient of each sequence in batch at a given lag.

    Args:
        x: Tensor of shape (batch_size, seq_len), binary or real-valued spike train.
        lag: time lag for autocorrelation.

    Returns:
        Tensor of shape (batch_size,) with autocorrelation coefficients.
    """
    # mean over time
    mu = x.mean(dim=1, keepdim=True)
    x_centered = x - mu

    # numerator: sum_{t=0 to T-lag-1} (x_t - mu)(x_{t+lag}-mu)
    num = (x_centered[:, :-lag] * x_centered[:, lag:]).sum(dim=1)
    # denominator: sum_{t=0 to T-1} (x_t - mu)^2
    den = (x_centered ** 2).sum(dim=1)

    # avoid division by zero
    eps = 1e-8
    return num / (den + eps)


class CBS(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(output_channel)
        self.active = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.active(x)

        return x


class Neurotransmitter(nn.Module):
    def __init__(self, transmitters: int = 100, neural: int = 1000, layers: int = 5, T: int = 16):
        super().__init__()

        self.layers = layers

        self.conv = nn.Sequential(
            nn.Conv1d(T, transmitters * 2, 15, stride=3),
            nn.BatchNorm1d(transmitters * 2),
            nn.SiLU()
        )

        self.block_list = nn.ModuleList()
        self.CBS_list = nn.ModuleList()
        self.CBS_cut_list = nn.ModuleList()
        for i in range(layers):
            block = nn.Sequential(
                nn.Conv1d(transmitters * 2, transmitters * 4, 3, stride=1, padding=1),
                nn.BatchNorm1d(transmitters * 4),
                nn.SiLU(),
                nn.Conv1d(transmitters * 4, transmitters * 2, 3, stride=1, padding=1),
                nn.BatchNorm1d(transmitters * 2),
                nn.SiLU(),
            )

            cbs = CBS(transmitters * 2, transmitters * 2, kernel_size=5, stride=2)
            cbs_cut = CBS(transmitters * 2, transmitters * 2, kernel_size=5, stride=2)

            self.block_list.append(block)
            self.CBS_list.append(cbs)
            self.CBS_cut_list.append(cbs_cut)

        self.fTransmitters = nn.Sequential(
            nn.Conv1d(transmitters * 2, transmitters, 15, stride=3),
            nn.BatchNorm1d(transmitters),
            nn.Sigmoid(),
        )

        self.effect = nn.Sequential(
            nn.Linear(transmitters, neural),
            nn.Dropout1d(p=0.5),
            nn.Tanh()
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        """

        Args:
            x: [T, bs, previous_neural]

        Returns: [bs, neural]

        """

        x = x.permute(1, 0, 2)
        x = self.conv(x)

        for i in range(self.layers):
            block = self.block_list[i]
            cbs = self.CBS_list[i]
            cbs_cut = self.CBS_cut_list[i]

            o1 = block(cbs(x))
            o2 = cbs_cut(x)
            x = o1 + o2

        x = self.global_max_pool(self.fTransmitters(x)).squeeze(-1)  # [bs, transmitters]

        return self.effect(x)


class HiddenNeurons(nn.Module):
    def __init__(self, previous_neural, neural, Neurotransmitter: nn.Module):
        super().__init__()

        self.get_pulse = layer.Linear(previous_neural, neural)
        self.neurotransmitter = Neurotransmitter
        self.neuron = nn.Sequential(
            neuron.LIFNode(
                tau=2.0,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode='m',
                detach_reset=True
            ),
            layer.SynapseFilter(  # 突触电流动态
                tau=2.,
                learnable=True,
                step_mode='m'
            )
        )

    def forward(self, x):
        """

        Args:
            x: [T, bs, previous_neural]

        Returns: [T, bs, neural]

        """

        x = self.get_pulse(x)
        effect = self.neurotransmitter(x)
        x = x * effect + x
        return self.neuron(x)


# ----------------------------------------
# 超神经回路模型
# ----------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, transmitters, transmitter_layers, T):
        super().__init__()

        self.T = T
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # 输入层
        self.encoder = encoding.PoissonEncoder(step_mode='m')
        self.input_neural = nn.Sequential(
            layer.Linear(input_size, hidden_size),
            neuron.LIFNode(
                tau=2.0,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode='m',
                detach_reset=True
            ),
        )

        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        self.neurotransmitter = Neurotransmitter(transmitters, hidden_size, transmitter_layers, T)
        for i in range(num_layers):
            self.hidden_layers.append(HiddenNeurons(hidden_size, hidden_size, self.neurotransmitter))

        # 输出层
        self.output_neural = nn.Sequential(
            layer.Linear(hidden_size, 1, bias=False),
            neuron.LIFNode(
                tau=2.0,
                v_threshold=1.0,
                v_reset=0.0,
                surrogate_function=surrogate.ATan(),
                step_mode='m',
                detach_reset=True
            ),
        )

    def firing_frequency_plot(self, data_list, T, base_height=2, height_per_layer=1):
        """
        绘制激发频率图，图高会随着隐藏层数动态增加，
        热图取值范围固定在 [0,1]，颜色条刻度在最右侧。

        参数：
        - data_list: list，包含输入层、若干隐藏层、输出层的张量列表
        - T: str，保存的文件名（不含后缀）
        - base_height: float，图像底部高度（用于输入层条形图部分）
        - height_per_layer: float，每个隐藏层额外增加的高度
        """
        # 计算平均值
        input_layer = data_list[0]  # [bs, hidden_size]
        avgbs_input_layer = input_layer.mean(dim=0).cpu().numpy()
        hidden_layers = torch.stack(data_list[1:-1], dim=0)  # [layers, bs, hidden_size]
        avgbs_hidden_layer = hidden_layers.mean(dim=1).cpu().numpy()  # [layers, hidden_size]

        num_layers, hidden_size = avgbs_hidden_layer.shape

        # 动态计算图像高度
        total_height = base_height + num_layers * height_per_layer
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, total_height),
            gridspec_kw={'height_ratios': [base_height, num_layers * height_per_layer]}
        )

        # —— 输入层条形图 —— #
        x = np.arange(len(avgbs_input_layer))
        ax1.bar(x, avgbs_input_layer, width=0.8, alpha=0.7, color='skyblue')
        ax1.set_title('输入层平均激发率 (按维度)')
        # 部分刻度
        step = max(1, len(x) // 10)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([str(i) for i in x[::step]], rotation=45, fontsize=8)
        ax1.set_ylabel('平均激发率')
        ax1.set_ylim(0, 1)  # 强制 y 轴范围 0~1

        # —— 隐藏层热图 —— #
        cax = ax2.imshow(
            avgbs_hidden_layer,
            cmap='viridis',
            aspect='auto',
            interpolation='nearest',
            vmin=0, vmax=1  # 统一取值范围 0~1
        )
        ax2.set_title('隐藏层平均激发率热图')
        ax2.set_ylabel('隐藏层层数')
        ax2.set_yticks(range(num_layers))
        ax2.set_yticklabels([f'Layer {i + 1}' for i in range(num_layers)])
        # 同样部分 x 刻度
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels([str(i) for i in x[::step]], rotation=45, fontsize=6)

        # 颜色条放在最右侧
        cb = fig.colorbar(
            cax,
            ax=[ax1, ax2],  # 共享颜色条
            orientation='vertical',
            pad=0.02,
            shrink=0.8
        )
        cb.set_label('平均激发率')
        cb.ax.yaxis.set_ticks_position('right')
        cb.ax.yaxis.set_label_position('right')

        plt.savefig(f'{T}.png', dpi=150)
        plt.close(fig)

    def forward(self, x):
        """

        Args:
            x: [bs, input_size]

        Returns: [bs, 1]

        """
        # 神经场发放状态
        firing_frequency_list = []

        x = x.unsqueeze(0).repeat(self.T, 1, 1)
        x = self.encoder(x)
        x = self.input_neural(x)  # [T, B, hidden-size]
        firing_frequency_list.append(x.mean(dim=0).detach())  # [B, hidden-size]]

        for layer in self.hidden_layers:
            x = layer(x)  # [T, B, hidden-size]
            firing_frequency_list.append(x.mean(dim=0).detach())  # [B, hidden-size]]

        x = self.output_neural(x).squeeze(-1).permute(1, 0)  # [T, B] -> [B, T]
        firing_frequency_list.append(x.mean(dim=-1).detach())  # [B]
        pos_p = autocorr_coefficient(x).unsqueeze(-1) + 1  # [0, 2]
        neg_p = - pos_p
        o = F.softmax(torch.cat((neg_p, pos_p), dim=-1), dim=-1)

        return o, firing_frequency_list


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    num_inputs = 8000
    hidden_dim = 1024
    num_layers = 4
    T = 4
    bs = 5

    model = NeuralNet(input_size=num_inputs,
                      hidden_size=hidden_dim,
                      num_layers=num_layers, transmitters=100,
                      transmitter_layers=3, T=T).to(device)

    total_params = count_model_params(model)
    trainable_params = count_trainable_params(model)

    print(f"模型的总参数量: {total_params}")
    print(f"模型的可训练参数量: {trainable_params}")

    dummy_input = torch.rand(T, bs, num_inputs).to(device)
    # 模型前向传播
    start_time = time.time()
    count = 0
    for input in dummy_input:
        output, firing_frequency_list = model(input)
        model.firing_frequency_plot(firing_frequency_list, count)
        count += 1
        print("输出形状:", output.shape)
    end_time = time.time()
    print("cost time: ", end_time - start_time)
