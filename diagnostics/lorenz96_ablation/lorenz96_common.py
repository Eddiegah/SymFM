"""
Extracted verbatim (structure-preserving) from notebooks/SymFM_Colab.ipynb
in github.com/Eddiegah/SymFM, cells 7, 11, 13, 15 -- the exact code that
produced the paper's Lorenz-96 SymFM results. Used here to run a genuine
component-wise ablation at N=4.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp


def lorenz96(t, x, F_forcing=8.0):
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F_forcing
    return dxdt


def generate_lorenz96_data(N, F=8.0, t_end=20.0, dt=0.01, n_trajectories=50,
                            noise_level=0.01, seed=42):
    np.random.seed(seed)
    t_eval = np.arange(0.0, t_end, dt)
    T = len(t_eval)
    X_all = np.zeros((n_trajectories, T, N))
    dXdt_all = np.zeros((n_trajectories, T, N))
    for traj_idx in range(n_trajectories):
        x0 = F * np.ones(N)
        x0[0] += 0.01 * np.random.randn()
        x0 += 0.5 * np.random.randn(N)
        sol = solve_ivp(lambda t, x: lorenz96(t, x, F_forcing=F),
                         (0.0, t_end), x0, method='RK45',
                         t_eval=t_eval, rtol=1e-8, atol=1e-8)
        if sol.success:
            X_traj = sol.y.T
            X_all[traj_idx] = X_traj
            dXdt_all[traj_idx] = np.array([lorenz96(t_eval[k], X_traj[k], F_forcing=F)
                                            for k in range(T)])
    return X_all, dXdt_all, t_eval


class ActiveSubspaceProjection(nn.Module):
    def __init__(self, N, d, eta=0.01):
        super().__init__()
        self.N = N
        self.d = d
        self.eta = eta
        W_init = torch.randn(d, N)
        Q, _ = torch.linalg.qr(W_init.T)
        self.W = nn.Parameter(Q[:, :d].T)

    def forward(self, x):
        return x @ self.W.T

    def orthonormality_loss(self):
        WWT = self.W @ self.W.T
        I_d = torch.eye(self.d, device=self.W.device)
        return self.eta * torch.norm(WWT - I_d, p='fro') ** 2


def lorenz96_residual(x, dxdt_pred, F_forcing=8.0):
    N = x.shape[1]
    dxdt_true = torch.zeros_like(x)
    for i in range(N):
        ip1 = (i + 1) % N
        im2 = (i - 2) % N
        im1 = (i - 1) % N
        dxdt_true[:, i] = (x[:, ip1] - x[:, im2]) * x[:, im1] - x[:, i] + F_forcing
    return F.mse_loss(dxdt_pred, dxdt_true)


class HierarchicalSymbolicHead(nn.Module):
    def __init__(self, d, N, hidden=256):
        super().__init__()
        self.d = d
        self.N = N
        self.univariate = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N)
        )
        self.pairwise = nn.Sequential(
            nn.Linear(d * d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N)
        )
        self.combine = nn.Linear(2 * N, N, bias=True)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        nn.init.zeros_(self.combine.bias)
        self.l1_weight = 0.001

    def forward(self, x_proj):
        uni = self.univariate(x_proj)
        batch = x_proj.shape[0]
        outer = (x_proj.unsqueeze(2) * x_proj.unsqueeze(1)).view(batch, -1)
        pair = self.pairwise(outer)
        return self.combine(torch.cat([uni, pair], dim=-1))

    def sparsity_loss(self):
        return self.l1_weight * torch.norm(self.combine.weight, p=1)


class SymFM(nn.Module):
    def __init__(self, N, d, hidden=256, eta=0.01):
        super().__init__()
        self.N = N
        self.d = d
        self.projection = ActiveSubspaceProjection(N=N, d=d, eta=eta)
        self.symbolic_head = HierarchicalSymbolicHead(d=d, N=N, hidden=hidden)

    def forward(self, x):
        x_proj = self.projection(x)
        dxdt_pred = self.symbolic_head(x_proj)
        return dxdt_pred, x_proj

    def compute_loss(self, x, dxdt_true, lambda1=1.0, lambda2=0.1,
                      lambda3=0.01, lambda4=5.0):
        dxdt_pred, x_proj = self.forward(x)
        l_rec = F.huber_loss(dxdt_pred, dxdt_true, delta=0.5)
        l_sparse = self.symbolic_head.sparsity_loss()
        l_ortho = self.projection.orthonormality_loss()
        l_physics = lorenz96_residual(x, dxdt_pred)
        loss = lambda1 * l_rec + lambda2 * l_sparse + lambda3 * l_ortho + lambda4 * l_physics
        return loss, {'total': float(loss.item()), 'rec': float(l_rec.item()),
                       'physics': float(l_physics.item())}


def train_symfm(X_train, dXdt_train, X_val, dXdt_val, N, d, n_epochs=5000,
                 lr=3e-4, lambda4=5.0, device='cpu', seed=0):
    torch.manual_seed(seed)
    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(dXdt_train, dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_vl = torch.tensor(dXdt_val, dtype=torch.float32).to(device)

    model = SymFM(N=N, d=d, hidden=256, eta=0.01).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=2, eta_min=1e-6
    )

    best_val = float('inf')
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss, ld = model.compute_loss(X_tr, y_tr, lambda4=lambda4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl, _ = model.compute_loss(X_vl, y_vl, lambda4=lambda4)
        if float(vl.item()) < best_val:
            best_val = float(vl.item())
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_symfm_model(model, X_test, dXdt_test, device='cpu', tol=0.15):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        dxdt_pred, _ = model.forward(X_t)
        pred_np = dxdt_pred.cpu().numpy()
    num = np.linalg.norm(dXdt_test - pred_np)
    den = np.linalg.norm(dXdt_test)
    l2 = float(num / (den + 1e-10))
    return {'recovered': l2 < tol, 'l2': min(l2, 10.0), 'raw_l2': l2}
