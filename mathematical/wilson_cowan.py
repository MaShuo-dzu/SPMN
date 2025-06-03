import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
import matplotlib.pyplot as plt


# ----------------------------------------
# Wilson-Cowan 群体方程的神经场实现层
# ----------------------------------------
class WilsonCowanNeuralField(nn.Module):
    def __init__(self, height, width, tau=10.0, dt=1.0, kernel_size=15, sigma_exc=2.0, sigma_inh=5.0):
        """
        构造函数：
        参数:
          height, width: 神经场空间尺寸
          tau: 神经元膜电位时间常数 τ（Wilson-Cowan微分方程中的时间尺度）
          dt: 时间步长 Δt
          kernel_size: 卷积核尺寸（用于模拟邻域连接）
          sigma_exc, sigma_inh: 高斯激励和抑制核的标准差，分别模拟兴奋和抑制连接的空间范围
        """
        super().__init__()
        self.height = height
        self.width = width
        self.tau = tau  # τ，控制神经场动力学速度
        self.dt = dt  # Δt，数值积分步长

        # 初始化空间耦合核 (W) 参数，代表神经场中的兴奋和抑制权重
        self.kernel = self._create_kernel(kernel_size, sigma_exc, sigma_inh)

    def _create_kernel(self, size, sigma_exc, sigma_inh):
        """
        构造空间耦合核 W(x,y) = Gaussian_exc(x,y) - 0.7 * Gaussian_inh(x,y)
        激励部分是空间范围较小的高斯核，
        抑制部分是空间范围较大的高斯核，两者叠加形成中心-抑制边缘环状效应。
        """
        x = torch.arange(size).float() - size // 2  # 坐标轴，中心对齐
        y = torch.arange(size).float() - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # 兴奋高斯核，σ较小
        gauss_exc = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_exc ** 2))
        # 抑制高斯核，σ较大
        gauss_inh = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_inh ** 2))

        kernel = gauss_exc - 0.7 * gauss_inh  # 加权叠加，0.7为抑制权重系数
        kernel = kernel / kernel.sum()  # 归一化确保权重和为1，稳定动态

        # 让卷积核成为神经网络参数，可以通过训练学习调整
        kernel_param = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))
        # unsqueeze拓展维度，适配conv2d权重格式 (out_channels, in_channels, H, W)
        return kernel_param

    def forward(self, u, I):
        """
        Wilson-Cowan群体动力学离散形式：
        du/dt = (-u + S(W * u + I)) / tau
        其中：
          u: 神经场当前活动状态，张量形状 [batch, 1, H, W]  此处用作记忆张量
          I: 外部输入刺激，形状同u  此处用作事件分析
          W: 空间耦合核，通过卷积实现神经元间局部连接影响
          S(): 激活函数，使用sigmoid模拟神经元非线性响应
          tau: 时间常数

        使用Euler显式法离散：
        u(t+Δt) = u(t) + Δt * du/dt
        """
        # 空间卷积，计算邻域激励和抑制影响 W * u
        conv = F.conv2d(u, self.kernel, padding=self.kernel.shape[-1] // 2)

        # 计算du/dt，sigmoid保证输出在0~1之间，模拟神经元激活函数
        du = (-u + torch.sigmoid(conv + I)) / self.tau

        # 时间步进更新神经场状态 u(t+Δt)
        u_next = u + self.dt * du

        return u_next


# ----------------------------------------
# 基于神经动力学的损失函数
# ----------------------------------------
class MemoryLoss(nn.Module):
    def __init__(self):
        """
        使用均方误差 (MSE) 作为记忆存储损失函数：
        Loss = MSE(u_final, memory_pattern)
        目标是使最终神经场状态 u_final 能够准确重现目标记忆模式 memory_pattern。
        """
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, u_final, memory_pattern):
        return self.mse_loss(u_final, memory_pattern)


# ----------------------------------------
# 生成目标记忆模式（高斯 bump）
# ----------------------------------------
def generate_gaussian_bump(batch_size, height, width, sigma=0.1, device='cpu'):
    """
    在二维空间中生成高斯形状的激活模式，模拟单个记忆的空间激活分布

    数学形式：
    M(x,y) = exp(-((x - x0)^2 + (y - y0)^2) / (2 * σ^2))
    这里假设中心在空间原点，x,y归一化为[-1,1]
    """
    x = torch.linspace(-1, 1, width, device=device)
    y = torch.linspace(-1, 1, height, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    bump = torch.exp(-((xx) ** 2 + (yy) ** 2) / (2 * sigma ** 2))

    # 扩展成batch形式，单通道格式，方便与u匹配
    bump = bump.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return bump


# ----------------------------------------
# 多步神经场动力学演化，并测速
# ----------------------------------------
def run_neural_field(u_init, I, wc_layer, steps=20):
    """
    输入：
      u_init: 初始神经场状态 u(0)
      I: 外部输入刺激 I
      wc_layer: Wilson-Cowan神经场层
      steps: 演化步数

    功能：
      对Wilson-Cowan动力学进行steps次迭代，模拟神经活动随时间变化过程。
      同时计时，返回所有状态历史和耗时。
    """
    start_time = time.time()
    u = u_init
    u_history = [u.detach().cpu().numpy()]  # 保存初始状态
    for _ in range(steps):
        u = wc_layer(u, I)
        u_history.append(u.detach().cpu().numpy())  # 记录每一步的状态
    duration = time.time() - start_time
    return u_history, duration


# 修改后的主程序，增加多步可视化
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    height, width = 64, 64
    steps = 30  # 动力学迭代步数

    # 初始化Wilson-Cowan神经场层
    wc_layer = WilsonCowanNeuralField(height, width).to(device)

    # 初始化神经场状态u(0)，小幅随机激活
    u_init = torch.rand(batch_size, 1, height, width, device=device) * 0.1

    # 生成目标记忆模式（空间高斯激活），作为“期望的记忆”
    memory_pattern = generate_gaussian_bump(batch_size, height, width, sigma=0.15, device=device)

    # 记忆输入刺激I，用目标记忆模式直接作为输入刺激（等效于外部提示）
    I = memory_pattern.clone()

    # 运行神经场演化，获取所有步骤的状态历史
    u_history, elapsed = run_neural_field(u_init, I, wc_layer, steps=steps)

    # 计算记忆损失
    loss_fn = MemoryLoss()
    loss = loss_fn(torch.tensor(u_history[-1]).to(device), memory_pattern)

    print(f"Neural field evolution took {elapsed:.4f} seconds")
    print(f"Memory loss (MSE) = {loss.item():.6f}")

    # 可视化多步变化
    idx = 0  # 展示第一个样本
    selected_steps = [0, 5, 10, 15, 20, 25, 30]  # 选择展示的步骤

    plt.figure(figsize=(20, 10))
    plt.suptitle("Neural Field Activity at Different Steps", fontsize=16)

    # 绘制每个选定步骤的活动状态
    for i, step in enumerate(selected_steps):
        plt.subplot(2, 4, i + 1)
        plt.imshow(u_history[step][idx, 0], cmap='hot')
        plt.title(f"Step {step}")
        plt.colorbar()

    # 添加目标模式对比
    plt.subplot(2, 4, 8)
    plt.imshow(memory_pattern.cpu()[idx, 0], cmap='hot')
    plt.title("Target Memory Pattern")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # 可选：动态变化动画（取消注释以启用）
    # fig = plt.figure()
    # ims = []
    # for step in range(steps+1):
    #     im = plt.imshow(u_history[step][idx, 0], cmap='hot', animated=True)
    #     plt.title(f"Step {step}")
    #     ims.append([im])
    # ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    # plt.show()