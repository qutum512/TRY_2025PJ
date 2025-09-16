# -*- coding: utf-8 -*-
"""
Lorenz + Transformer (Encoder–Decoder, Seq2Seq)
- Positions: Sinusoidal PE
- Standardize for training; de-standardize for plotting
- Train with teacher forcing (BOS + shifted targets)
- Inference: autoregressively generate K steps
- Figures & metrics style matches previous linear/encoder-only script
"""

import os
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Lorenz system
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
T = 5000            # total steps
dt = 0.01           # time step
x0, y0, z0 = 1.0, 1.0, 1.0

# Data & training
seq_len = 128       # encoder history length L
future_steps = 50   # K:target horizon
epochs = 800
batch_size = 128
learning_rate = 5e-4
weight_decay = 0.0

# Transformer
d_model = 128
nhead = 4
num_layers = 2

# Output directory (same style as your previous script)
result_root = "/SSD_DISK/users/qiutian/project_set/results"



# =============== Utilities ===============
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def lorenz_system(x0, y0, z0, sigma, rho, beta, steps, dt):
    xs = np.zeros(steps, dtype=np.float32)
    ys = np.zeros(steps, dtype=np.float32)
    zs = np.zeros(steps, dtype=np.float32)
    xs[0], ys[0], zs[0] = x0, y0, z0
    for i in range(1, steps):
        dx = sigma * (ys[i-1] - xs[i-1])
        dy = xs[i-1] * (rho - zs[i-1]) - ys[i-1]
        dz = xs[i-1] * ys[i-1] - beta * zs[i-1]
        xs[i] = xs[i-1] + dx * dt
        ys[i] = ys[i-1] + dy * dt
        zs[i] = zs[i-1] + dz * dt
    return xs, ys, zs


# =============== Positional Encoding ===============
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B,T,d_model)
        return x + self.pe[:, :x.size(1), :]


# =============== EncDec Model (predict K steps) ===============
class EncDecNextK(nn.Module):
    """
    Encoder input:  (B, L, 2)   history (x,y) normalized
    Decoder target: (B, K, 2)   future (x,y) normalized
    Output:         (B, K, 2)   predicted future (normalized)
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, input_dim=2):
        super().__init__()
        # Encoder
        self.enc_in = nn.Linear(input_dim, d_model)
        self.enc_pe = SinusoidalPE(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # Decoder
        self.dec_in = nn.Linear(input_dim, d_model)
        self.dec_pe = SinusoidalPE(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        # Head
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        """
        src: (B,L,2) normalized history
        tgt: (B,K,2) decoder inputs (teacher forcing inputs: BOS + target[:-1])
        """
        # Encoder
        h = self.enc_in(src)
        h = self.enc_pe(h)
        mem = self.encoder(h)                        # (B,L,d)

        # Decoder with causal mask
        K = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=tgt.device), diagonal=1
        )  # True = mask future
        g = self.dec_in(tgt)
        g = self.dec_pe(g)
        out = self.decoder(g, mem, tgt_mask=tgt_mask)  # (B,K,d)
        yhat = self.head(out)                           # (B,K,2)
        return yhat

    @staticmethod
    def make_tf_inputs(y_target):
        """
        Teacher forcing inputs = [BOS(zeros)] + target[:, :-1, :]
        y_target: (B,K,2) normalized
        return:   (B,K,2)
        """
        B, K, C = y_target.shape
        bos = torch.zeros(B, 1, C, device=y_target.device, dtype=y_target.dtype)
        return torch.cat([bos, y_target[:, :-1, :]], dim=1)


def build_seq2seq_dataset(data_n, L: int, K: int):
    X, Y = [], []
    Tn = len(data_n)
    for i in range(Tn - L - K):
        X.append(data_n[i:i+L])       # (L,2)
        Y.append(data_n[i+L:i+L+K])   # (K,2)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    return X, Y


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ===== 1) Generate Lorenz data =====
    xs, ys, zs = lorenz_system(x0, y0, z0, sigma, rho, beta, T, dt)
    data = np.stack([xs, ys], axis=1).astype(np.float32)  # (T,2)

    # ===== 2) Standardize & build (L -> K) pairs =====
    mean = data.mean(axis=0, keepdims=True)
    std  = data.std(axis=0, keepdims=True) + 1e-8
    data_n = (data - mean) / std

    X, Y = build_seq2seq_dataset(data_n, seq_len, future_steps)  # (N,L,2),(N,K,2)
    # ===== 2.5) Train / Val split =====
    num_pairs = X.shape[0]
    split = int(0.8 * num_pairs)
    Xtr, Ytr = X[:split], Y[:split]
    Xva, Yva = X[split:], Y[split:]
    Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)

    # ===== 3) Model, optimizer, sched, helpers =====
    model = EncDecNextK(d_model=d_model, nhead=nhead, num_layers=num_layers, input_dim=2).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, verbose=True)

    def batch_iter(Xt, Yt, bs):
        idx = torch.randperm(Xt.size(0), device=Xt.device)
        for i in range(0, len(idx), bs):
            j = idx[i:i+bs]
            yield Xt[j], Yt[j]

    @torch.no_grad()
    def eval_val_loss(model, Xv, Yv, bs=512):
        model.eval()
        total, cnt = 0.0, 0
        for xb, yb in batch_iter(Xv, Yv, bs):
            dec_in = model.make_tf_inputs(yb)
            pred = model(xb, dec_in)
            total += loss_fn(pred, yb).item() * xb.size(0)
            cnt += xb.size(0)
        return total / max(cnt, 1)

    # ===== 3.1) Training with validation & early stopping =====
    best_val = float('inf')
    patience, bad_epochs = 30, 0
    ckpt_dir = os.path.join(result_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best_encdec.pt")

    for ep in range(1, epochs + 1):
        model.train()
        total_train, seen = 0.0, 0
        for xb, yb in batch_iter(Xtr, Ytr, batch_size):
            dec_in = model.make_tf_inputs(yb)     # (B,K,2)
            pred = model(xb, dec_in)              # (B,K,2)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_train += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = total_train / max(seen, 1)
        val_loss = eval_val_loss(model, Xva, Yva, bs=512)
        sched.step(val_loss)

        if ep % 10 == 0 or ep == 1:
            print(f"[Epoch {ep:04d}] train {train_loss:.6f} | val {val_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

        # early stopping & ckpt
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] no val improvement for {patience} epochs. Best val={best_val:.6f}")
                break

    # 训练结束后：加载 best checkpoint
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Load Best] val={best_val:.6f} from {ckpt_path}")


    # ===== 4) Inference (AR generate K steps, normalized domain) =====
    model.eval()
    with torch.no_grad():
        # history seed = last L points from standardized data
        hist_seed_n = torch.tensor(data_n[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)  # (1,L,2)
        # true future (for evaluation) by integrating from true last state:
        x0_f, y0_f, z0_f = xs[-1], ys[-1], zs[-1]
        xs_f, ys_f, zs_f = lorenz_system(x0_f, y0_f, z0_f, sigma, rho, beta, steps=future_steps+1, dt=dt)
        true_future_xy = np.stack([xs_f[1:], ys_f[1:]], axis=1)   # (K,2)

        # AR decoding
        preds_n = []
        prev = torch.zeros(1, 1, 2, device=device)   # BOS (zeros)
        for step in range(future_steps):
            # build current decoder input of length (step+1)
            dec_in_step = prev  # (1, step+1, 2)
            # pad to K for a fixed-shape forward (causal mask handles future)
            if dec_in_step.size(1) < future_steps:
                pad = torch.zeros(1, future_steps - dec_in_step.size(1), 2, device=device)
                dec_in_full = torch.cat([dec_in_step, pad], dim=1)  # (1,K,2)
            else:
                dec_in_full = dec_in_step

            pred_full = model(hist_seed_n, dec_in_full)  # (1,K,2)
            y_next = pred_full[:, dec_in_step.size(1)-1, :].unsqueeze(1)  # (1,1,2)
            preds_n.append(y_next.squeeze(0).squeeze(0).cpu().numpy())
            prev = torch.cat([prev, y_next], dim=1)

        preds_n = np.stack(preds_n, axis=0)             # (K,2)

    # De-standardize
    pred_future = preds_n * std + mean                  # (K,2)

    # ===== 5) Visualization & evaluation (same style) =====
    t_hist = np.arange(T - seq_len, T) * dt
    t_future = np.arange(T, T + future_steps) * dt
    hist_seed = data[-seq_len:]  # (L,2)

    residual = pred_future - true_future_xy
    resid_x, resid_y = residual[:, 0], residual[:, 1]

    # Output dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(result_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Figures will be saved to: {save_dir}")

    # Fig1: time series
    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(t_hist, hist_seed[:, 0], label='History x', linewidth=1.3)
    ax1.plot(t_hist, hist_seed[:, 1], label='History y', linewidth=1.3)
    ax1.plot(t_future, true_future_xy[:, 0], label='True future x', linewidth=1.5)
    ax1.plot(t_future, true_future_xy[:, 1], label='True future y', linewidth=1.5)
    ax1.plot(t_future, pred_future[:, 0], '--', label='Predicted x', linewidth=1.5)
    ax1.plot(t_future, pred_future[:, 1], '--', label='Predicted y', linewidth=1.5)
    ax1.axvline(t_hist[-1], color='k', linestyle='--', linewidth=1.0)
    ax1.axvspan(t_hist[0], t_hist[-1], color='gray', alpha=0.08, label='History range')
    ax1.axvspan(t_future[0], t_future[-1], color='orange', alpha=0.05, label='Future range')
    ax1.set_title('Lorenz: True vs Predicted (EncDec, horizon K)')
    ax1.set_xlabel('Time t'); ax1.set_ylabel('Value')
    ax1.legend(ncol=3, fontsize=9); ax1.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "figure1_timeseries.png"), dpi=300)
    plt.close(fig)

    # Fig2: phase plot
    fig2 = plt.figure(figsize=(6, 5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(hist_seed[:, 0], hist_seed[:, 1], alpha=0.9, label='History (x,y)')
    ax2.plot(true_future_xy[:, 0], true_future_xy[:, 1], alpha=0.9, label='True future (x,y)')
    ax2.plot(pred_future[:, 0], pred_future[:, 1], '--', alpha=0.9, label='Predicted (x,y)')
    ax2.set_title('Phase plot: x-y')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.legend(); ax2.grid(alpha=0.2)
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "figure2_phase.png"), dpi=300)
    plt.close(fig2)

    # Fig3: residual over time
    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(t_future, resid_x, label='Residual x', linewidth=1.2)
    ax3.plot(t_future, resid_y, label='Residual y', linewidth=1.2)
    ax3.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    ax3.set_title('Residual over time')
    ax3.set_xlabel('Time t'); ax3.set_ylabel('Residual')
    ax3.legend(); ax3.grid(alpha=0.2)
    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, "figure3_residual.png"), dpi=300)
    plt.close(fig3)

    # Fig4: residual hist & pred-vs-true scatter
    fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(resid_x, bins=60, alpha=0.7, density=True, label='Residual x')
    axes[0].hist(resid_y, bins=60, alpha=0.7, density=True, label='Residual y')
    axes[0].set_title('Residual histogram'); axes[0].set_xlabel('Residual'); axes[0].set_ylabel('Density')
    axes[0].legend(); axes[0].grid(alpha=0.2)

    axes[1].scatter(true_future_xy[:, 0], pred_future[:, 0], s=10, alpha=0.7, label='x: Pred vs True')
    axes[1].scatter(true_future_xy[:, 1], pred_future[:, 1], s=10, alpha=0.7, label='y: Pred vs True')
    min_v = float(np.min([true_future_xy.min(), pred_future.min()]))
    max_v = float(np.max([true_future_xy.max(), pred_future.max()]))
    axes[1].plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1, label='y = x')
    axes[1].set_title('Scatter: Prediction vs Truth')
    axes[1].set_xlabel('True'); axes[1].set_ylabel('Predicted')
    axes[1].legend(); axes[1].grid(alpha=0.2)

    plt.tight_layout()
    fig4.savefig(os.path.join(save_dir, "figure4_hist_scatter.png"), dpi=300)
    plt.close(fig4)

    # ===== Metrics =====
    rmse_x = float(np.sqrt(np.mean(resid_x**2)))
    rmse_y = float(np.sqrt(np.mean(resid_y**2)))
    mae_x = float(np.mean(np.abs(resid_x)))
    mae_y = float(np.mean(np.abs(resid_y)))

    print(f"[EVAL] RMSE  x: {rmse_x:.6f} | y: {rmse_y:.6f}")
    print(f"[EVAL] MAE   x: {mae_x:.6f} | y: {mae_y:.6f}")
    print(f"[INFO] History length: {seq_len}, Future steps: {future_steps}, dt={dt}")
    print(f"[DONE] All figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
