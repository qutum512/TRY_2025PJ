# -*- coding: utf-8 -*-
"""
Encoder+Decoder Transformer：用过去 L 点预测未来 K 点
- Encoder: 双向 self-attn 读取历史窗口
- Decoder: 因果 masked self-attn + cross-attn 到 Encoder 上下文
- Train: teacher forcing（<BOS> + 真实前缀） -> 监督 K 个未来点
- Inference: 自回归生成未来 K 点
- 保存结果到 /SSD_DISK/users/qiutian/project_set/results/<timestamp>/
"""

import os, math, json, datetime
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ===================== 0) 超参数 =====================
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/SSD_DISK/users/qiutian/project_set/results"

N = 5000       # 全序列长度
dt = 0.01
omega = 5*math.pi
L = 128        # 历史窗口长度（Encoder 输入）
K = 50         # 未来长度（Decoder 目标/输出）

d_model = 64
nhead = 4
num_layers = 2
epochs = 300
batch_size = 256
lr = 5e-4

# ===================== 1) 数据 =====================
t = torch.arange(N, dtype=torch.float32, device=device) * dt
x = torch.cos(omega * t)  # [N]

def make_seq2seq_dataset(series: torch.Tensor, L: int, K: int):
    X_enc, Y_dec = [], []
    for i in range(len(series) - L - K):
        hist = series[i:i+L]           # encoder 输入
        fut  = series[i+L:i+L+K]       # decoder 目标
        X_enc.append(hist)
        Y_dec.append(fut)
    X = torch.stack(X_enc, 0).unsqueeze(-1)  # [M, L, 1]
    Y = torch.stack(Y_dec, 0).unsqueeze(-1)  # [M, K, 1]
    return X, Y

X, Y = make_seq2seq_dataset(x, L, K)
split = int(0.8 * X.size(0))
Xtr, Ytr = X[:split], Y[:split]
Xva, Yva = X[split:], Y[split:]
Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)

# ===================== 2) 模型 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):  # x:[B,T,d]
        return x + self.pe[:, :x.size(1), :]

def causal_tgt_mask(T: int, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

class EncDecTS(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # encoder
        self.enc_in = nn.Linear(1, d_model)
        self.enc_pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # decoder
        self.dec_in = nn.Linear(1, d_model)
        self.dec_pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, src, tgt):
        h = self.enc_pos(self.enc_in(src))         # [B,L,d]
        mem = self.encoder(h)                      # [B,L,d]
        T = tgt.size(1)
        mask = causal_tgt_mask(T, tgt.device)
        g = self.dec_pos(self.dec_in(tgt))
        out = self.decoder(g, mem, tgt_mask=mask)  # [B,K,d]
        return self.head(out)                      # [B,K,1]
    def teacher_forcing_inputs(self, y_target):
        B, K, _ = y_target.shape
        bos_val = torch.zeros(B, 1, 1, device=y_target.device)
        return torch.cat([bos_val, y_target[:, :-1, :]], dim=1)

model = EncDecTS(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ===================== 3) 训练 =====================
def batch_iter(Xt, Yt, bs):
    idx = torch.randperm(Xt.size(0), device= Xt.device)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield Xt[j], Yt[j]

for ep in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    for xb, yb in batch_iter(Xtr, Ytr, batch_size):
        opt.zero_grad()
        dec_in = model.teacher_forcing_inputs(yb)
        yhat = model(xb, dec_in)
        loss = loss_fn(yhat, yb)
        loss.backward()
        opt.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= Xtr.size(0)
    model.eval()
    with torch.no_grad():
        dec_in = model.teacher_forcing_inputs(Yva)
        val_pred = model(Xva, dec_in)
        val_loss = loss_fn(val_pred, Yva).item()
    if ep % 20 == 0 or ep == 1 or ep == epochs:
        print(f"epoch {ep:3d} | train {train_loss:.3e} | val {val_loss:.3e}")

# ===================== 4) 推理 =====================
model.eval()
with torch.no_grad():
    src_hist = x[-(L+K):-K].unsqueeze(0).unsqueeze(-1)   # [1,L,1]
    tgt_true = x[-K:].unsqueeze(0).unsqueeze(-1)         # [1,K,1]
    preds = []
    dec_in = torch.zeros(1, 1, 1, device=device)         # 初始<BOS>
    for step in range(K):
        dec_full = torch.zeros(1, K, 1, device=device)
        dec_full[:, :dec_in.size(1), :] = dec_in
        yhat_full = model(src_hist, dec_full)            # [1,K,1]
        y_next = yhat_full[:, dec_in.size(1)-1, :].unsqueeze(1)
        dec_in = torch.cat([dec_in, y_next], dim=1)
        preds.append(y_next)
    y_pred = torch.cat(preds, dim=1).squeeze(0).squeeze(-1).cpu()
    y_true = tgt_true.squeeze(0).squeeze(-1).cpu()

# ===================== 5) 保存结果 =====================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(BASE_DIR, timestamp)
os.makedirs(outdir, exist_ok=True)

# 拆出“历史 L 点 + 未来 K 点”
hist_t = t[-(L+K):-K].detach().cpu()
hist_x = x[-(L+K):-K].detach().cpu()
fut_t  = t[-K:].detach().cpu()
y_pred = y_pred.detach().cpu()
y_true = y_true.detach().cpu()

resid = (y_pred - y_true)
mse = (resid**2).mean().item()
print(f"Seq2Seq rollout MSE over {K} steps = {mse:.3e}")

fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=False)

# 顶部：历史 + 未来 同图展示
axs[0].plot(hist_t.numpy(), hist_x.numpy(), label="history (encoder input)")
axs[0].plot(fut_t.numpy(),  y_true.numpy(), label="true future")
axs[0].plot(fut_t.numpy(),  y_pred.numpy(), label="EncDec prediction", alpha=0.85)

# 正确的 warmup 阴影：画在“历史段”的最后 L 个点
axs[0].axvspan(hist_t[-L].item(), hist_t[-1].item(),
               color='gray', alpha=0.12, label="warmup region (history)")

axs[0].legend(loc="upper right")
axs[0].set_ylabel("value")

# 底部：未来残差
axs[1].plot(fut_t.numpy(), resid.numpy(), label="residual (pred - true)")
axs[1].axhline(0, color="k", linestyle="--", linewidth=1)
axs[1].legend(loc="upper right")
axs[1].set_xlabel("t")
axs[1].set_ylabel("residual")

plt.tight_layout()
fig.savefig(os.path.join(outdir, "prediction.png"), dpi=220)
plt.close(fig)

# 保存 tensor 与配置
torch.save(y_pred, os.path.join(outdir, "y_pred.pt"))
torch.save(y_true, os.path.join(outdir, "y_true.pt"))
config = {
    "N": N, "dt": dt, "omega": omega, "L": L, "K": K,
    "d_model": d_model, "nhead": nhead, "num_layers": num_layers,
    "epochs": epochs, "batch_size": batch_size, "lr": lr,
    "device": str(device), "mse_rollout_K": mse
}
with open(os.path.join(outdir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"[Saved] {outdir}")
print("[Saved] prediction.png, y_pred.pt, y_true.pt, config.json")
