import os, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =================== HYPERPARAMETERS ===================
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
T = 5000
dt = 0.01
x0, y0, z0 = 1.0, 1.0, 1.0

epochs = 4000
lr = 3e-3
batch_size = 256
seq_len = 128
standardize_xy = True
lam_coef = 1e-6
seed = 42

mlp_hidden = 512
mlp_layers = 4

future_steps = 50
result_root = "/SSD_DISK/users/qiutian/project_set/results"

# =======================================================

def set_seed(s=42):
    np.random.seed(s); random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def lorenz(x0, y0, z0, sigma, rho, beta, steps, dt):
    xs = np.zeros(steps, dtype=np.float32)
    ys = np.zeros(steps, dtype=np.float32)
    zs = np.zeros(steps, dtype=np.float32)
    xs[0], ys[0], zs[0] = x0, y0, z0
    for i in range(1, steps):
        dx = sigma * (ys[i-1] - xs[i-1])
        dy = xs[i-1]*(rho - zs[i-1]) - ys[i-1]
        dz = xs[i-1]*ys[i-1] - beta*zs[i-1]
        xs[i] = xs[i-1] + dx*dt
        ys[i] = ys[i-1] + dy*dt
        zs[i] = zs[i-1] + dz*dt
    return xs, ys, zs

# =================== Dataset ===================
def build_dataset_xy_windows(data_xy_n, seq_len, dt):
    Ttot = data_xy_n.shape[0]
    i_start = seq_len - 1
    i_end = Ttot - 2
    idxs = np.arange(i_start, i_end+1, dtype=np.int64)
    N = len(idxs)

    X = np.zeros((N, 2*seq_len), dtype=np.float32)
    y_deriv = np.zeros((N, 2), dtype=np.float32)

    x = data_xy_n[:,0]; y = data_xy_n[:,1]
    for k, i in enumerate(idxs):
        win = data_xy_n[i-seq_len+1:i+1, :]
        X[k] = win.reshape(-1)
        dx_i = (x[i+1] - x[i-1]) / (2.0*dt)
        dy_i = (y[i+1] - y[i-1]) / (2.0*dt)
        y_deriv[k,0] = dx_i
        y_deriv[k,1] = dy_i

    tail = data_xy_n[-seq_len:, :].T.reshape(1, 2, seq_len).astype(np.float32)
    xy_tail_for_rollout = torch.from_numpy(tail)
    return X, y_deriv, xy_tail_for_rollout, idxs

def rk4_step(x_t, y_t, z_t, coeff, dt):
    def fxy(x, y, z):
        one = torch.ones_like(x)
        x2, y2, z2 = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        phi = torch.cat([one, x, y, z, x2, y2, z2, xy, xz, yz], dim=1)
        fx = (phi * coeff.cx.view(1, -1)).sum(dim=1, keepdim=True)
        fy = (phi * coeff.cy.view(1, -1)).sum(dim=1, keepdim=True)
        return fx, fy
    k1x, k1y = fxy(x_t, y_t, z_t)
    k2x, k2y = fxy(x_t + 0.5*dt*k1x, y_t + 0.5*dt*k1y, z_t)
    k3x, k3y = fxy(x_t + 0.5*dt*k2x, y_t + 0.5*dt*k2y, z_t)
    k4x, k4y = fxy(x_t + dt*k3x,     y_t + dt*k3y,     z_t)
    x_next = x_t + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    y_next = y_t + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
    return x_next, y_next

# =================== Model ===================
class MLP_z_only(nn.Module):
    def __init__(self, seq_len, hidden=256, layers=3):
        super().__init__()
        in_dim = 2 * seq_len
        h = hidden
        blocks = [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
        for _ in range(layers-1):
            blocks += [nn.Linear(h, h), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*blocks)
        self.head_z = nn.Linear(h, 1)
    def forward(self, x_flat):
        h = self.net(x_flat)
        return self.head_z(h)

class GlobalQuadCoeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.cx = nn.Parameter(torch.zeros(10))
        self.cy = nn.Parameter(torch.zeros(10))

# =================== Utils ===================
def phi_from_xyzt(x_t, y_t, z_t):
    one = torch.ones_like(x_t)
    x2, y2, z2 = x_t*x_t, y_t*y_t, z_t*z_t
    xy, xz, yz = x_t*y_t, x_t*z_t, y_t*z_t
    return torch.cat([one, x_t, y_t, z_t, x2, y2, z2, xy, xz, yz], dim=1)

@torch.no_grad()
def rollout_future_mlp_global(xy_hist_n, model_z, coeff, dt, steps, device):
    cur = xy_hist_n.clone().to(device)
    preds = []
    for _ in range(steps):
        x_t = cur[:,0,-1:].view(1,1)
        y_t = cur[:,1,-1:].view(1,1)
        x_flat = cur.permute(0,2,1).reshape(1,-1)   #修正
        z_hat = model_z(x_flat)
        x_next, y_next = rk4_step(x_t, y_t, z_hat, coeff, dt)
        preds.append(torch.cat([x_next,y_next],1).squeeze(0).cpu().numpy())
        nxt = torch.cat([x_next,y_next],0).view(1,2,1)
        cur = torch.cat([cur,nxt],-1)[:,:, -cur.size(-1):]
    return np.stack(preds,0)

# =================== Main ===================
def main():
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    xs,ys,zs = lorenz(x0,y0,z0,sigma,rho,beta,T,dt)
    data_xy = np.stack([xs,ys],1).astype(np.float32)
    if standardize_xy:
        mean_xy = data_xy.mean(0,keepdims=True)
        std_xy = data_xy.std(0,keepdims=True)+1e-8
        data_xy_n = (data_xy-mean_xy)/std_xy
    else:
        mean_xy = np.zeros((1,2),np.float32)
        std_xy = np.ones((1,2),np.float32)
        data_xy_n = data_xy

    X,y_deriv,xy_tail_rollout,idxs = build_dataset_xy_windows(data_xy_n,seq_len,dt)
    N = X.shape[0]
    print(f"[INFO] dataset windows={N}, window length={seq_len}")
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y_deriv).to(device)
    xy_tail_rollout = xy_tail_rollout.to(device)

    model_z = MLP_z_only(seq_len,hidden=mlp_hidden,layers=mlp_layers).to(device)
    coeff = GlobalQuadCoeff().to(device)
    opt = optim.Adam(list(model_z.parameters())+list(coeff.parameters()), lr=lr)

    # ---- Training ----
    num_batches = int(np.ceil(N/batch_size))
    for ep in range(1,epochs+1):
        perm = torch.randperm(N,device=device)
        tot_loss=tot_x=tot_y=tot_l2=0.0
        for b in range(num_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            xb,yb = X_t[idx],y_t[idx]
            x_last,y_last = xb[:,-2],xb[:,-1]
            z_hat = model_z(xb)
            phi = phi_from_xyzt(x_last.view(-1,1), y_last.view(-1,1), z_hat)
            rhs_x = (phi*coeff.cx.view(1,-1)).sum(1,keepdim=True)
            rhs_y = (phi*coeff.cy.view(1,-1)).sum(1,keepdim=True)
            dx_tgt,dy_tgt = yb[:,0:1], yb[:,1:2]
            loss_x=((dx_tgt-rhs_x)**2).mean()
            loss_y=((dy_tgt-rhs_y)**2).mean()
            loss_l2=lam_coef*(coeff.cx.pow(2).mean()+coeff.cy.pow(2).mean())
            loss=loss_x+loss_y+loss_l2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(model_z.parameters())+list(coeff.parameters()),1.0)
            opt.step()
            tot_loss+=loss.item(); tot_x+=loss_x.item(); tot_y+=loss_y.item(); tot_l2+=loss_l2.item()
        if ep%100==0 or ep==1:
            print(f"[{ep:04d}] loss={tot_loss/num_batches:.6f} | "
                  f"loss_x={tot_x/num_batches:.6f} | loss_y={tot_y/num_batches:.6f} | "
                  f"l2={tot_l2/num_batches:.6f}")

    # ---- Print coeffs ----
    basis=["1","x","y","z","x^2","y^2","z^2","xy","xz","yz"]
    cx_np,cy_np=coeff.cx.detach().cpu().numpy(),coeff.cy.detach().cpu().numpy()
    print("\n[LEARNED GLOBAL COEFFS]")
    for i,b in enumerate(basis):
        print(f" c_x[{b:>2}]={cx_np[i]: .6f}   c_y[{b:>2}]={cy_np[i]: .6f}")

    # ---- Rollout ----
    model_z.eval()
    with torch.no_grad():
        preds_n=rollout_future_mlp_global(xy_tail_rollout,model_z,coeff,dt,future_steps,device)
    preds_phys=preds_n*std_xy+mean_xy

    x0f,y0f,z0f=xs[-1],ys[-1],zs[-1]
    xs_f,ys_f,zs_f=lorenz(x0f,y0f,z0f,sigma,rho,beta,future_steps+1,dt)
    true_future=np.stack([xs_f[1:],ys_f[1:]],1)
    resid=preds_phys-true_future
    rmse_x,rmse_y=np.sqrt((resid[:,0]**2).mean()),np.sqrt((resid[:,1]**2).mean())
    mae_x,mae_y=np.abs(resid[:,0]).mean(),np.abs(resid[:,1]).mean()

    ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir=os.path.join(result_root,f"mlp_global_coeff_rollout_{ts}")
    os.makedirs(save_dir,exist_ok=True)
    print(f"\n[SAVE] {save_dir}")
    print(f"[ROLL-OUT METRICS] RMSE x:{rmse_x:.6f} y:{rmse_y:.6f} | MAE x:{mae_x:.6f} y:{mae_y:.6f}")

    # ---- Visualization ----
    t_hist=np.arange(T-seq_len,T)*dt
    t_future=np.arange(T,T+future_steps)*dt
    hist_xy_phys=data_xy[-seq_len:,:]

    # 1) timeseries
    fig=plt.figure(figsize=(13,7)); ax=fig.add_subplot(111)
    ax.plot(t_hist,hist_xy_phys[:,0],label="History x")
    ax.plot(t_hist,hist_xy_phys[:,1],label="History y")
    ax.plot(t_future,true_future[:,0],label="True future x")
    ax.plot(t_future,true_future[:,1],label="True future y")
    ax.plot(t_future,preds_phys[:,0],'--',label="Pred x")
    ax.plot(t_future,preds_phys[:,1],'--',label="Pred y")
    ax.axvline(t_hist[-1],c='k',ls='--'); ax.legend(); ax.grid()
    fig.savefig(os.path.join(save_dir,"figure1_timeseries.png"),dpi=300); plt.close(fig)

    # 2) phase
    fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111)
    ax.plot(hist_xy_phys[:,0],hist_xy_phys[:,1],label="History")
    ax.plot(true_future[:,0],true_future[:,1],label="True")
    ax.plot(preds_phys[:,0],preds_phys[:,1],'--',label="Pred")
    ax.legend(); ax.grid()
    fig.savefig(os.path.join(save_dir,"figure2_phase.png"),dpi=300); plt.close(fig)

    # 3) residual
    fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111)
    ax.plot(t_future,resid[:,0],label="Resid x")
    ax.plot(t_future,resid[:,1],label="Resid y")
    ax.axhline(0,c='k',ls='--'); ax.legend(); ax.grid()
    fig.savefig(os.path.join(save_dir,"figure3_residual.png"),dpi=300); plt.close(fig)

    # 4) hist+scatter
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].hist(resid[:,0],bins=60,alpha=0.7,label="Resid x")
    axes[0].hist(resid[:,1],bins=60,alpha=0.7,label="Resid y")
    axes[0].legend(); axes[0].grid()
    axes[1].scatter(true_future[:,0],preds_phys[:,0],s=10,alpha=0.7,label="x")
    axes[1].scatter(true_future[:,1],preds_phys[:,1],s=10,alpha=0.7,label="y")
    minv,maxv=min(true_future.min(),preds_phys.min()),max(true_future.max(),preds_phys.max())
    axes[1].plot([minv,maxv],[minv,maxv],'k--'); axes[1].legend(); axes[1].grid()
    fig.savefig(os.path.join(save_dir,"figure4_hist_scatter.png"),dpi=300); plt.close(fig)

    print(f"[DONE] Figures & coeffs saved to {save_dir}")

if __name__=="__main__":
    main()

