import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ===============================
#  连续两时间尺度系统 (简谐振动)
#  生成模拟数据：慢变量 μ(t) + 粒子独立扩散噪声
# ===============================
class ContinuousSystem:
    def __init__(self):
        # 粒子数量和时间步
        self.N_particles = 1000
        self.steps = 1000
        self.diffusion_sigma = 0.5  # 粒子扩散噪声标准差

        # 慢变量的简谐振动参数
        self.sho_amplitude = 5.0
        self.sho_frequency = 0.01

        # 训练参数
        self.train_deltas = [1, 2, 4, 8, 16]  # 训练中预测的时间间隔
        self.base_delta = 2
        self.k_list = (2, 3)  # 半群一致性检查用的倍数

        # 模型参数
        self.dim_z = 1      # 潜变量维度
        self.lr = 1e-3
        self.epochs = 200
        self.batch_pred = 128   # bootstrap预测损失的batch size
        self.batch_semi = 128   # 半群一致性损失的batch size

    def simulate(self):
        """
        模拟粒子轨迹：
        - 每个粒子围绕 μ(t) 高斯噪声扩散
        - 返回 snapshots (粒子快变量矩阵) 和 μ(t) (慢变量)
        """
        ts = np.arange(self.steps + 1)
        mu_traj = self.sho_amplitude * np.sin(self.sho_frequency * ts)  # 慢变量 μ(t)

        noise = np.random.normal(0, self.diffusion_sigma, size=(self.steps + 1, self.N_particles))
        snapshots = mu_traj[:, np.newaxis] + noise  # 每个粒子加噪声

        # 转为torch tensor
        return torch.tensor(snapshots, dtype=torch.float32), torch.tensor(mu_traj, dtype=torch.float32)

# ===============================
# 编码器 & 动力学模型
#  Encoder: 将粒子快变量映射为潜变量 z
#  PhiDelta: 学习 z 的时间演化
# ===============================
class Encoder(nn.Module):
    def __init__(self, dim_z=1):
        super().__init__()
        # MLP结构
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, dim_z)
        )

    def forward(self, x):
        """
        输入: x [batch, N_particles]
        输出: z [batch, dim_z]
        - 先计算分位数，压缩粒子分布信息
        - 再通过MLP映射到潜变量
        """
        q = torch.quantile(x, torch.linspace(0.05, 0.95, 20, device=x.device), dim=1).T
        return self.net(q)

class PhiDelta(nn.Module):
    def __init__(self, dim_z=1):
        super().__init__()
        # MLP结构
        self.net = nn.Sequential(
            nn.Linear(dim_z + 1, 32),
            nn.ReLU(),
            nn.Linear(32, dim_z)
        )

    def forward(self, z, delta):
        """
        输入: z [batch, dim_z], delta [batch]
        输出: z(t+delta)
        - 将 z 和 log(delta) 拼接后输入 MLP，学习潜变量演化
        - 返回 z + dz 实现增量预测
        """
        delta = delta.view(-1, 1)
        dz = self.net(torch.cat([z, torch.log(delta + 1e-6)], dim=-1))
        return z + dz

# ===============================
# 训练器
# BootstrapTrainer: 负责计算损失、训练和评估
# ===============================
class BootstrapTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # 初始化模型
        self.encoder = Encoder(config.dim_z).to(device)
        self.phi = PhiDelta(config.dim_z).to(device)

        # Adam优化器
        self.opt = optim.Adam(list(self.encoder.parameters()) + list(self.phi.parameters()), lr=config.lr)

        # 存储每次评估的仿射变换参数
        self.affine_params = []

    # -----------------------------
    # one step 预测损失
    # -----------------------------
    def bootstrap_prediction_loss(self, snapshots):
        steps_plus_1 = snapshots.shape[0]
        t_list, d_list = [], []

        # 随机选 batch_pred 个时间点及对应Δ
        for _ in range(self.config.batch_pred):
            Δ = np.random.choice(self.config.train_deltas)
            t = np.random.randint(0, steps_plus_1 - Δ)
            t_list.append(t)
            d_list.append(Δ)

        idx_t = torch.tensor(t_list, device=self.device, dtype=torch.long)
        idx_td = torch.tensor([t + d for t, d in zip(t_list, d_list)], device=self.device, dtype=torch.long)

        x_t = snapshots[idx_t]      # 当前时间粒子快变量
        x_td = snapshots[idx_td]    # Δ时间后的粒子快变量

        z_t = self.encoder(x_t)     # 编码器得到潜变量 z(t)
        with torch.no_grad():
            z_td = self.encoder(x_td)   # z(t+Δ) (不更新梯度)

        delta_tensor = torch.tensor(d_list, dtype=torch.float32, device=self.device)
        z_pred = self.phi(z_t, delta_tensor)  # 预测 z(t+Δ)
        return F.mse_loss(z_pred, z_td)       # 均方误差

    # -----------------------------
    # bootstrap一致性损失
    # -----------------------------
    def bootstrap_consistency_loss(self, snapshots):
        steps_plus_1 = snapshots.shape[0]
        losses = []
        for k in self.config.k_list:
            Δ1 = self.config.base_delta
            Δk = self.config.base_delta * k

            t = torch.randint(0, steps_plus_1 - Δk, (self.config.batch_semi,), device=self.device)
            x = snapshots[t]
            z = self.encoder(x)

            # 逐步应用 Δ1 k 次
            z_comp = z.clone()
            for _ in range(k):
                z_comp = self.phi(z_comp, torch.full((self.config.batch_semi,), float(Δ1), device=self.device))

            # 直接应用 Δk
            z_k = self.phi(z, torch.full((self.config.batch_semi,), float(Δk), device=self.device))
            losses.append(F.mse_loss(z_comp, z_k))
        return sum(losses) / len(losses)

    # -----------------------------
    # 单个epoch训练
    # -----------------------------
    def train_epoch(self, snapshots):
        self.opt.zero_grad()
        L_pred = self.bootstrap_prediction_loss(snapshots)
        L_semi = self.bootstrap_consistency_loss(snapshots)
        loss = L_pred + 0.5 * L_semi     # 总loss
        loss.backward()                  # 反向传播计算梯度
        self.opt.step()                  # 参数更新
        return loss.item(), L_pred.item(), L_semi.item()

    # -----------------------------
    # 仿射变换拟合 z -> μ(t)
    # -----------------------------
    def fit_affine_transformation(self, z, H_true):
        """
        线性最小二乘拟合:
        z_aligned = a * z + b
        """
        A = torch.stack([z, torch.ones_like(z)], dim=1)
        solution = torch.linalg.lstsq(A, H_true.unsqueeze(1)).solution
        a, b = solution[0, 0], solution[1, 0]
        return a, b

    # -----------------------------
    # 评估
    # -----------------------------
    def evaluate(self, snapshots, mu_traj):
        with torch.no_grad():
            z = self.encoder(snapshots.to(self.device)).cpu().squeeze(-1)
        a, b = self.fit_affine_transformation(z, mu_traj)
        z_aligned = a * z + b
        self.affine_params.append((a.item(), b.item()))
        return mu_traj, z_aligned, z, a, b

# ===============================
# 主函数
# ===============================
def main():
    config = ContinuousSystem()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = BootstrapTrainer(config, device)

    print("开始训练...")
    for epoch in range(config.epochs):
        # 生成一段新的模拟数据
        snapshots, mu_traj = config.simulate()
        snapshots = snapshots.to(device)

        # 训练
        loss, L_pred, L_semi = trainer.train_epoch(snapshots)

        # 每20个epoch打印一次训练信息
        if epoch % 20 == 0:
            print(f"epoch {epoch:3d} | loss={loss:.4f} | pred={L_pred:.4f} | semi={L_semi:.4f}")

    print("训练结束！")

    # -----------------------------
    # 多次评估
    # -----------------------------
    num_evaluations = 5
    all_results = []

    for i in range(num_evaluations):
        print(f"\n评估运行 {i+1}/{num_evaluations}:")
        snapshots, mu_traj = config.simulate()
        mu_true, z_aligned, z_raw, a, b = trainer.evaluate(snapshots, mu_traj)
        all_results.append((mu_true, z_aligned, z_raw, a, b))

    # 绘制最后一次评估结果
    mu_true, z_aligned, z_raw, a, b = all_results[-1]
    ts = np.arange(config.steps + 1)

    plt.figure(figsize=(12, 8))

    # 上图: 对齐后的潜变量 vs μ(t)
    plt.subplot(2, 1, 1)
    plt.plot(ts, mu_true, label="True μ(t) (Sine Wave)", linewidth=2, color='blue')
    plt.plot(ts, z_aligned, "--", label=f"Aligned z (a={a:.2f}, b={b:.2f})", linewidth=2, color='red')
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.title("Bootstrap learning of slow oscillation with affine transformation")
    plt.grid(True, alpha=0.3)

    # 下图: 未对齐的潜变量 z_raw
    plt.subplot(2, 1, 2)
    plt.plot(ts, z_raw, label="Raw z (before affine transformation)", alpha=0.7, color='green')
    plt.xlabel("time step")
    plt.ylabel("latent value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

