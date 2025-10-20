# utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(path, model, optimizer, epoch, val_rollout_loss, is_best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_rollout_loss': val_rollout_loss,
    }
    torch.save(ckpt, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "checkpoint_best.pth")
        torch.save(ckpt, best_path)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    start_epoch = ckpt.get('epoch', 0) + 1
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    val_rollout_loss = ckpt.get('val_rollout_loss', None)
    return model, optimizer, start_epoch, val_rollout_loss

def per_step_rollout_mse(model, dataset_windows, device, max_steps=None):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for xb, yb in dataset_windows:
            xb = xb.to(device)
            yb = yb.to(device)
            Tf = yb.size(1)
            if max_steps is not None:
                Tf = min(Tf, max_steps)
            preds = model.rollout(xb, steps=Tf)
            se = ((preds - yb[:, :Tf, :]) ** 2).mean(dim=0).squeeze(-1).cpu().numpy()
            all_errors.append(se)
    all_errors = np.stack(all_errors, axis=0)
    mse_mean = all_errors.mean(axis=0)
    mse_std = all_errors.std(axis=0)
    return mse_mean, mse_std

def plot_mse_curve(mse_mean, mse_std=None, title='per-step MSE', savepath=None):
    xs = np.arange(len(mse_mean))
    plt.figure(figsize=(6,3))
    plt.plot(xs, mse_mean, label='MSE')
    if mse_std is not None:
        plt.fill_between(xs, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2)
    plt.yscale('log')
    plt.xlabel('forecast step')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
