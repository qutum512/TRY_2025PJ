# config.py
from dataclasses import dataclass
import os
import torch

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Data generation
    dt: float = 0.01
    n_traj_train: int = 400
    n_steps_per_traj: int = 2000

    # Eval rollout on a long single sequence
    n_steps_eval: int = 768
    future_len_eval: int = 512

    # Windowing
    past_len: int = 256
    future_len_train: int = 256
    stride: int = 1
    train_split: float = 0.9

    # Model
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    clip_grad: float = 1.0

    # Output
    plot: bool = True
    result_dir: str = "/home/Product set/result"

cfg = Config()
os.makedirs(cfg.result_dir, exist_ok=True)
