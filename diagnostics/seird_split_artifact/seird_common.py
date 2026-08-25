"""
Extracted verbatim (structure-preserving) from notebooks/SymFM_NS_SEIRD_Colab.ipynb
in github.com/Eddiegah/SymFM, cells 4, 10, 12, 14 -- the exact code that produced
the paper's "SEIRD SymFM Results: N=50/250/500: ERR=0.0% L2=10.0000" finding.
Used here to re-run and diagnose that result on CPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------- Model (cell-4)
class ActiveSubspaceProjection(nn.Module):
    def __init__(self, N, d, eta=0.01):
        super().__init__()
        self.N = N; self.d = d; self.eta = eta
        W_init = torch.randn(d, N)
        Q, _ = torch.linalg.qr(W_init.T)
        self.W = nn.Parameter(Q[:, :d].T)
    def forward(self, x): return x @ self.W.T
    def reconstruct(self, x_proj): return x_proj @ self.W
    def orthonormality_loss(self):
        WWT = self.W @ self.W.T
        return self.eta * torch.norm(WWT - torch.eye(self.d, device=self.W.device), p='fro')**2

class HierarchicalSymbolicHead(nn.Module):
    def __init__(self, d, N, hidden=256):
        super().__init__()
        self.d = d; self.N = N
        self.univariate = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, N))
        self.pairwise = nn.Sequential(
            nn.Linear(d*d, hidden), nn.SiLU(),
            nn.Linear(hidden, N))
        self.combine = nn.Linear(2*N, N, bias=True)
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        nn.init.zeros_(self.combine.bias)
    def forward(self, x_proj):
        uni = self.univariate(x_proj)
        b = x_proj.shape[0]
        outer = (x_proj.unsqueeze(2) * x_proj.unsqueeze(1)).view(b, -1)
        pair = self.pairwise(outer)
        return self.combine(torch.cat([uni, pair], dim=-1))
    def sparsity_loss(self):
        return 0.001 * torch.norm(self.combine.weight, p=1)

class SymFM(nn.Module):
    def __init__(self, N, d, hidden=256, eta=0.01):
        super().__init__()
        self.N = N; self.d = d
        self.projection = ActiveSubspaceProjection(N=N, d=d, eta=eta)
        self.symbolic_head = HierarchicalSymbolicHead(d=d, N=N, hidden=hidden)
    def forward(self, x):
        x_proj = self.projection(x)
        return self.symbolic_head(x_proj), x_proj
    def compute_loss(self, x, y_true, physics_fn=None,
                     l1=1.0, l2=0.1, l3=0.01, l4=5.0):
        pred, xp = self.forward(x)
        l_rec = F.huber_loss(pred, y_true, delta=0.5)
        l_sp  = self.symbolic_head.sparsity_loss()
        l_ort = self.projection.orthonormality_loss()
        if physics_fn is not None:
            l_phy = physics_fn(x, pred)
        else:
            l_phy = torch.tensor(0.0, device=x.device)
        loss = l1*l_rec + l2*l_sp + l3*l_ort + l4*l_phy
        return loss, {'total': float(loss.item()), 'rec': float(l_rec.item()),
                      'physics': float(l_phy.item())}

def train_model(X_tr, y_tr, X_vl, y_vl, N, d, n_epochs=3000,
                lr=3e-4, physics_fn=None, device='cpu', verbose_every=None,
                seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_vl, dtype=torch.float32).to(device)
    yv = torch.tensor(y_vl, dtype=torch.float32).to(device)
    model = SymFM(N=N, d=d, hidden=256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1000, eta_min=1e-6)
    best_val = float('inf'); best_state = None
    history = []
    for epoch in range(n_epochs):
        model.train(); opt.zero_grad()
        loss, ld = model.compute_loss(Xt, yt, physics_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        model.eval()
        with torch.no_grad():
            vl, _ = model.compute_loss(Xv, yv, physics_fn)
        if float(vl.item()) < best_val:
            best_val = float(vl.item())
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        if verbose_every and (epoch+1) % verbose_every == 0:
            print(f'  Epoch {epoch+1}/{n_epochs} Train:{ld["total"]:.4f} Val:{vl.item():.4f} Rec:{ld["rec"]:.4f}')
            history.append((epoch+1, ld['total'], float(vl.item())))
    if best_state: model.load_state_dict(best_state)
    return model, history

def eval_model(model, X_test, y_test, device='cpu', tol=0.25):
    model.eval()
    with torch.no_grad():
        pred, _ = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        pred_np = pred.cpu().numpy()
    raw_l2 = float(np.linalg.norm(y_test - pred_np) / (np.linalg.norm(y_test) + 1e-10))
    return {'recovered': raw_l2 < tol, 'l2': min(raw_l2, 10.0), 'raw_l2': raw_l2,
            'pred_has_nan': bool(np.isnan(pred_np).any()),
            'pred_scale': float(np.abs(pred_np).mean()),
            'target_scale': float(np.abs(y_test).mean())}


# ---------------------------------------------------------------- SEIRD data (cell-10)
def simulate_seird(P, n_steps=300, dt=1.0, n_trajectories=15, seed=42):
    np.random.seed(seed)
    N_pop  = 1e6
    N_state = 5 * P

    def seird_rhs(t, y, beta, sigma, gamma, delta, kappa):
        S = y[0*P:1*P]; E = y[1*P:2*P]
        I = y[2*P:3*P]; R = y[3*P:4*P]; D = y[4*P:5*P]
        dS = np.zeros(P); dE = np.zeros(P)
        dI = np.zeros(P); dR = np.zeros(P); dD = np.zeros(P)
        for p in range(P):
            N_p  = S[p] + E[p] + I[p] + R[p]
            inf  = beta[p] * S[p] * I[p] / (N_p + 1e-6)
            coup = sum(kappa[p,q]*(S[q]-S[p]) for q in range(P) if q!=p)
            dS[p] = -inf + coup
            dE[p] =  inf - sigma[p] * E[p]
            dI[p] =  sigma[p]*E[p] - (gamma[p]+delta[p])*I[p]
            dR[p] =  gamma[p] * I[p]
            dD[p] =  delta[p] * I[p]
        return np.concatenate([dS,dE,dI,dR,dD])

    X_all    = np.zeros((n_trajectories, n_steps, N_state))
    dXdt_all = np.zeros((n_trajectories, n_steps, N_state))
    t_eval   = np.arange(n_steps) * dt

    print(f'Simulating SEIRD: P={P} patches, N={N_state}, {n_trajectories} trajectories')

    params_per_traj = []
    for traj in range(n_trajectories):
        beta  = np.random.uniform(0.2, 0.5, P)
        sigma = np.random.uniform(0.1, 0.2, P)
        gamma = np.random.uniform(0.05, 0.15, P)
        delta = np.random.uniform(0.005, 0.02, P)
        kappa = np.random.uniform(0.0, 0.01, (P,P))
        np.fill_diagonal(kappa, 0)
        params_per_traj.append((beta, sigma, gamma, delta, kappa))

        S0 = N_pop * np.ones(P)
        E0 = np.random.uniform(10, 100, P)
        I0 = np.random.uniform(5, 50, P)
        R0 = np.zeros(P); D0 = np.zeros(P)
        y0 = np.concatenate([S0,E0,I0,R0,D0])

        sol = solve_ivp(
            lambda t,y: seird_rhs(t,y,beta,sigma,gamma,delta,kappa),
            (0, n_steps*dt), y0, method='RK45',
            t_eval=t_eval, rtol=1e-6, atol=1e-6
        )

        if sol.success:
            X_traj = sol.y.T
            X_traj = X_traj / N_pop
            X_all[traj]    = X_traj
            for k in range(n_steps):
                dXdt_all[traj,k] = seird_rhs(t_eval[k], X_traj[k]*N_pop,
                                              beta,sigma,gamma,delta,kappa) / N_pop

        if (traj+1) % 5 == 0:
            print(f'  {traj+1}/{n_trajectories} done')

    print(f'SEIRD data generation complete: N={N_state}')
    return X_all, dXdt_all, N_state, params_per_traj
