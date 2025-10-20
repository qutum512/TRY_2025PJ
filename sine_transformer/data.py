# data.py
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_sine(N, dt, omega, amplitude=1.0, phase=0.0, noise_std=0.0):
    t = np.arange(N, dtype=np.float64) * dt
    x = amplitude * np.sin(omega * t + phase)
    if noise_std > 0:
        x = x + np.random.normal(scale=noise_std, size=x.shape)
    return x[:, None].astype(np.float64), t  # [N,1]

def generate_multiple_sines(n_traj, N_per_traj, dt, noise_std=0.0):
    all_X = []
    for _ in range(n_traj):
        amp = np.random.uniform(0.5, 1.5)
        omega = np.random.uniform(0.5, 2.5)  # rad/s
        phase = np.random.uniform(0.0, 2 * math.pi)
        X, _ = generate_sine(N_per_traj, dt, omega=omega, amplitude=amp, phase=phase, noise_std=noise_std)
        all_X.append(X)
    return np.stack(all_X, axis=0)  # [K,N,1]

def build_windows(series, past_len, future_len, stride):
    N = series.shape[0]
    M = (N - past_len - future_len) // stride + 1
    src = np.zeros((M, past_len, 1), dtype=np.float32)
    tgt = np.zeros((M, future_len, 1), dtype=np.float32)
    i = 0
    for start in range(0, N - past_len - future_len + 1, stride):
        src[i] = series[start:start + past_len]
        tgt[i] = series[start + past_len:start + past_len + future_len]
        i += 1
    return src, tgt

def split_trajs(num_traj, train_ratio=0.9, seed=42):
    ids = np.arange(num_traj)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr = int(num_traj * train_ratio)
    return ids[:n_tr], ids[n_tr:]

def build_windows_for_idxs(X_all, idxs, mean_x, std_x, past_len, future_len, stride):
    src_ls, tgt_ls = [], []
    for i in idxs:
        series = X_all[i, :, :]
        series_n = (series - mean_x) / std_x
        s, t = build_windows(series_n, past_len, future_len, stride)
        src_ls.append(s)
        tgt_ls.append(t)
    return np.concatenate(src_ls, axis=0), np.concatenate(tgt_ls, axis=0)

class SeqDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = torch.from_numpy(src)
        self.tgt = torch.from_numpy(tgt)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

def make_loaders(src_tr, tgt_tr, src_va, tgt_va, cfg):
    train_loader = DataLoader(SeqDataset(src_tr, tgt_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(SeqDataset(src_va, tgt_va), batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader
