# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import cfg
from data import set_seed, generate_multiple_sines, split_trajs, build_windows_for_idxs, make_loaders
from model import TransformerSeq
from utils import save_checkpoint, load_checkpoint
import numpy as np

set_seed(cfg.seed)

# prepare data (same pipeline as original file)
all_train = generate_multiple_sines(cfg.n_traj_train, cfg.n_steps_per_traj, cfg.dt, noise_std=0.0)
train_ids, val_ids = split_trajs(all_train.shape[0], train_ratio=cfg.train_split, seed=cfg.seed)

train_flat = all_train[train_ids, :, :].reshape(-1, 1)
mean_x = train_flat.mean(axis=0, keepdims=True)
std_x = train_flat.std(axis=0, keepdims=True) + 1e-8
print(f"Train mean: {mean_x}, std: {std_x}")

src_tr, tgt_tr = build_windows_for_idxs(all_train, train_ids, mean_x, std_x, cfg.past_len, cfg.future_len_train, cfg.stride)
src_va, tgt_va = build_windows_for_idxs(all_train, val_ids, mean_x, std_x, cfg.past_len, cfg.future_len_train, cfg.stride)
print(f"Windows -> train: {src_tr.shape[0]}, val: {src_va.shape[0]}")

train_loader, val_loader = make_loaders(src_tr, tgt_tr, src_va, tgt_va, cfg)

model = TransformerSeq(
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_encoder_layers=cfg.num_encoder_layers,
    num_decoder_layers=cfg.num_decoder_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
).to(cfg.device)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
loss_fn = nn.MSELoss()

best_val_roll = float('inf')
start_epoch = 1
ckpt_path = os.path.join(cfg.result_dir, "checkpoint_last.pth")

def train_one_epoch():
    model.train()
    total, n = 0.0, 0
    for src, tgt in train_loader:
        src = src.to(cfg.device).float()
        tgt = tgt.to(cfg.device).float()

        start_tok = src[:, -1:, :]
        tgt_in = torch.cat([start_tok, tgt[:, :-1, :]], dim=1)

        pred = model(src, tgt_in)
        loss = loss_fn(pred, tgt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        optimizer.step()

        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch_teacher():
    model.eval()
    total, n = 0.0, 0
    for src, tgt in val_loader:
        src = src.to(cfg.device).float()
        tgt = tgt.to(cfg.device).float()
        start_tok = src[:, -1:, :]
        tgt_in = torch.cat([start_tok, tgt[:, :-1, :]], dim=1)
        pred = model(src, tgt_in)
        loss = loss_fn(pred, tgt)
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch_rollout():
    model.eval()
    total, n = 0.0, 0
    for src, tgt in val_loader:
        src = src.to(cfg.device).float()
        tgt = tgt.to(cfg.device).float()
        pred = model.rollout(src, steps=tgt.size(1))
        loss = loss_fn(pred, tgt)
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(n, 1)

print(f"Device: {cfg.device}")
print(f"Starting training for {cfg.epochs} epochs...")
for epoch in range(start_epoch, cfg.epochs + 1):
    tr_loss = train_one_epoch()
    va_tf = eval_one_epoch_teacher()
    va_roll = eval_one_epoch_rollout()
    scheduler.step()
    is_best = va_roll < best_val_roll
    if is_best:
        best_val_roll = va_roll
    save_checkpoint(os.path.join(cfg.result_dir, "checkpoint_last.pth"), model, optimizer, epoch, va_roll, is_best=is_best)
    print(f"[Epoch {epoch:02d}] train TF: {tr_loss:.6f} | val TF: {va_tf:.6f} | val rollout: {va_roll:.6f}")

# save normalization stats for eval
np.save(os.path.join(cfg.result_dir, "mean_x.npy"), mean_x)
np.save(os.path.join(cfg.result_dir, "std_x.npy"), std_x)
