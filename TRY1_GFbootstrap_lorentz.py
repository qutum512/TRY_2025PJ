import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.optim as optim
import random

sigma = 10.0
r = 28.0
b = 8/3

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return [dx, dy, dz]

# 初值
state0 = [1.0, 1.0, 1.0]

# 时间积分
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 50000)
sol = solve_ivp(lorenz, t_span, state0, t_eval=t_eval)

x, y, z = sol.y
t = sol.t


def delay_embed(x, k=7, tau=5):
    """
    构造延迟嵌入向量
    输入: x(t) 序列
    输出: N(k+1) 的延迟矩阵
    """
    N = len(x) - k*tau
    X = np.zeros((N, k+1))
    for i in range(N):
        X[i] = [x[i + j*tau] for j in range(k+1)]
    return X

k, tau = 7, 5
X_embed = delay_embed(x, k, tau) #size = (49965,8)
Y_target = np.stack([x[k*tau:], y[k*tau:], z[k*tau:]], axis=1) #size = (49965,3)

X_train = torch.tensor(X_embed, dtype=torch.float32)
Y_train = torch.tensor(Y_target, dtype=torch.float32)




class ReconNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)  # 输出新的一组x,y,z
        )
    def forward(self, x):
        return self.net(x)

model = ReconNet(input_dim=X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

## 定义mask,每个k*tau单元格有一个点进行检验点
N = X_train.shape[0]
mask = torch.zeros(N, dtype=torch.bool)
mask[::k*tau] = True

loss_x_weight = 1.0
loss_yz_weight = 1.0
loss_boot_weight = 100.0

for epoch in range(10000):
    optimizer.zero_grad()
    pred = model(X_train)
    ## x维度完全监督
    loss_x = loss_fn(pred[:, 0], Y_train[:, 0])
    ## y,z维度部分监督，区间端点 
    loss_yz = loss_fn(pred[mask][:,1:], Y_train[mask][:,1:])


    ## ktau 单元格的bootstrap
    interval = k*tau
    min_unit = tau
    num_intervals = N//interval
    possible_delta = [k*tau for k in range(1,interval//tau)]
  
    loss_boot_total = 0.0
    for i in range(num_intervals):
        start = i*interval
        end = start+interval

        X_block = X_train[start:end+1]
        Y_block = Y_train[start:end+1]
        pred_block = pred[start:end+1]

        #bootstrap的半群一致性  
        delta = random.choice([d for d in possible_delta if 2*d <= interval])

        x0_embed = X_block[0:1]
        f_delta = model(x0_embed)

        idx = 2*delta
        if idx >= X_block.shape[0]:
            idx = X_block.shape[0]-1
        x_input_2delta = X_block[idx:idx+1]

        #双步自洽方程
        f_2delta = model(x_input_2delta)
        f_target_2delta = 0.5*f_delta[:,1:] + 0.5*model(X_block[delta:delta+1]+delta*f_delta[:,0:1])[:,1:]

        loss_boot = loss_fn(f_2delta[:,1:],f_target_2delta)
        loss_boot_total += loss_boot
    
    loss_boot_avg = loss_boot_total / num_intervals


    total_loss = (loss_x_weight * loss_x +
                  loss_yz_weight * loss_yz +
                  loss_boot_weight * loss_boot_avg)

    total_loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Total {total_loss.item():.6f}, "
              f"Loss_x {loss_x.item():.6f}, Loss_yz {loss_yz.item():.6f}, "
              f"Loss_bootstrap {loss_boot_avg.item():.6f}")


Y_pred = model(X_train).detach().numpy()













# 可视化
plt.figure(figsize=(12,5))
plt.subplot(2,1,1)
plt.plot(t[k*tau:], Y_target[:,0], label="True y", color="g")
plt.plot(t[k*tau:], Y_pred[:,0], label="Pred y", color="g", linestyle="--")
plt.legend()
plt.title("Reconstruction of y(t) from x(t)")

plt.subplot(2,1,2)
plt.plot(t[k*tau:], Y_target[:,1], label="True z", color="b")
plt.plot(t[k*tau:], Y_pred[:,1], label="Pred z", color="b", linestyle="--")
plt.legend()
plt.title("Reconstruction of z(t) from x(t)")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x[k*tau:], Y_target[:,0], Y_target[:,1], color="black", lw=0.5)
ax1.set_title("True Lorenz Attractor (x,y,z)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x[k*tau:], Y_pred[:,0], Y_pred[:,1], color="red", lw=0.5)
ax2.set_title("Reconstructed Attractor from x(t)")

plt.show()


residual_y = Y_target[:,0] - Y_pred[:,0]
residual_z = Y_target[:,1] - Y_pred[:,1]

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.hist(residual_y, bins=100, color="g", alpha=0.7, edgecolor="black")
plt.title("Residual Histogram: y_true - y_pred")
plt.xlabel("Residual (y)")
plt.ylabel("Frequency")

plt.subplot(2,1,2)
plt.hist(residual_z, bins=100, color="b", alpha=0.7, edgecolor="black")
plt.title("Residual Histogram: z_true - z_pred")
plt.xlabel("Residual (z)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()