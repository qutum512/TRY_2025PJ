import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


N = 1000
dt = 0.01
t = torch.arange(N, dtype=torch.float32) * dt
omega = 2.0
x = torch.cos(omega * t)  

class YRegressor(nn.Module):
    def __init__(self, W=9, hidden=32):
        super().__init__()
        self.W = W
        self.conv = nn.Conv1d(1, W, kernel_size=W, stride=1, padding=W//2, bias=False)
        with torch.no_grad():
            eye = torch.eye(W).unsqueeze(1)
            self.conv.weight.copy_(eye)
        self.conv.weight.requires_grad_(False)

        self.net = nn.Sequential(
            nn.Linear(W, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.c11 = nn.Parameter(torch.tensor(0.0))
        self.c12 = nn.Parameter(torch.tensor(1.0))
        self.c21 = nn.Parameter(torch.tensor(-1.0))
        self.c22 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_seq):
        x_in = x_seq.unsqueeze(0).unsqueeze(0)  
        Xw = self.conv(x_in).squeeze(0).T       
        y_hat = self.net(Xw).squeeze(-1)       
        C = torch.stack([torch.stack([self.c11, self.c12]),
                         torch.stack([self.c21, self.c22])])
        return y_hat, C

def finite_diff(x, dt):
    dx = torch.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2*dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx

model = YRegressor(W=9)
opt = optim.Adam(model.parameters(), lr=1e-3)

for it in range(8000):
    opt.zero_grad()
    y_hat, C = model(x)
    c11, c12, c21, c22 = C[0,0], C[0,1], C[1,0], C[1,1]

    dx = finite_diff(x, dt)
    Lx = torch.mean((dx - (c11*x + c12*y_hat))**2)

    ydot_pred = c21*x[:-1] + c22*y_hat[:-1]
    ydot_fd   = (y_hat[1:] - y_hat[:-1]) / dt
    Ly = torch.mean((ydot_fd - ydot_pred)**2)

    L = Lx + Ly
    L.backward()
    opt.step()

    if (it+1) % 200 == 0:
        detC = (c11*c22 - c12*c21).item()
        print(f"iter {it+1}: loss={L.item():.4e}, det={detC:.4f}")

with torch.no_grad():
    y_hat, C = model(x)

print("Learned C ≈\n", C.detach().numpy())


with torch.no_grad():
    c11, c12, c21, c22 = C[0,0], C[0,1], C[1,0], C[1,1]
    resid = y_hat[1:] - y_hat[:-1] - dt*(c21*x[:-1] + c22*y_hat[:-1])


fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

axs[0].plot(t.numpy(), x.numpy(), label="x(t)")
axs[0].plot(t.numpy(), y_hat.numpy(), label="y_hat(t)")
axs[0].legend()
axs[0].set_ylabel("x, y_hat")

axs[1].plot(t[1:].numpy(), resid.numpy(), label="y residual")
axs[1].axhline(0, color="k", linestyle="--")
axs[1].legend()
axs[1].set_xlabel("t")
axs[1].set_ylabel("residual")

plt.tight_layout()
plt.show()

