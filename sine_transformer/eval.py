# eval.py
import os
import numpy as np
import torch
from config import cfg
from model import TransformerSeq
from utils import load_checkpoint
from data import generate_sine
import matplotlib.pyplot as plt

# load model
model = TransformerSeq(
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_encoder_layers=cfg.num_encoder_layers,
    num_decoder_layers=cfg.num_decoder_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
).to(cfg.device)

ckpt_path = os.path.join(cfg.result_dir, "checkpoint_last.pth")
if os.path.exists(ckpt_path):
    model, _, start_epoch, _ = load_checkpoint(ckpt_path, model, optimizer=None, device=cfg.device)
    print(f"Loaded checkpoint, resume epoch {start_epoch}")
else:
    print("No checkpoint found at", ckpt_path)

mean_x = np.load(os.path.join(cfg.result_dir, "mean_x.npy"))
std_x = np.load(os.path.join(cfg.result_dir, "std_x.npy"))

# create unseen sine
omega_test = 1.7
amp_test = 1.1
phase_test = 0.6
data_eval, t_eval = generate_sine(cfg.n_steps_eval, cfg.dt, omega=omega_test, amplitude=amp_test, phase=phase_test)
data_eval_n = (data_eval - mean_x) / std_x

past_start = 0
past_end = cfg.past_len
future_end = cfg.past_len + cfg.future_len_eval

with torch.no_grad():
    src_eval = torch.from_numpy(data_eval_n[past_start:past_end]).float().unsqueeze(0).to(cfg.device)
    tgt_true = torch.from_numpy(data_eval_n[past_end:future_end]).float().unsqueeze(0).to(cfg.device)
    preds_n = model.rollout(src_eval, steps=cfg.future_len_eval)

preds = preds_n.cpu().numpy() * std_x + mean_x
truth = tgt_true.cpu().numpy() * std_x + mean_x
time_future = t_eval[past_end:future_end]

rmse = float(np.sqrt(np.mean((preds[0, :, 0] - truth[0, :, 0]) ** 2)))
print(f"Rollout RMSE over {cfg.future_len_eval * cfg.dt:.2f}s: {rmse:.6f}")

if cfg.plot:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_future, truth[0, :, 0], label="true", linewidth=1.0)
    ax.plot(time_future, preds[0, :, 0], label="pred", linewidth=1.0, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x(t)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(cfg.result_dir, "sine_transformer_rollout.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")
