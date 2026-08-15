"""
symfm_model.py
--------------
Full SymFM model: combines active subspace projection with
a physics-informed hierarchical symbolic regression head.

This version uses:
- 2000 training epochs
- Learning rate 5e-4
- Physics-informed loss (Lorenz-96 residual penalty)
- Better initialisation and regularisation

Author: Edmund Eric Gah
Project: SymFM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from active_subspace import ActiveSubspaceProjection


# ── Physics-informed loss: Lorenz-96 residual ─────────────────────────────────

def lorenz96_residual(x, dxdt_pred, F_forcing=8.0):
    """
    Computes how much the predicted derivatives violate
    the Lorenz-96 governing equation structure.

    True equation: dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F

    Parameters
    ----------
    x          : torch.Tensor, shape (batch, N)
    dxdt_pred  : torch.Tensor, shape (batch, N)
    F_forcing  : float, Lorenz-96 forcing constant

    Returns
    -------
    residual_loss : torch.Tensor, scalar
    """
    N = x.shape[1]
    dxdt_true_approx = torch.zeros_like(x)
    for i in range(N):
        ip1 = (i + 1) % N
        im2 = (i - 2) % N
        im1 = (i - 1) % N
        dxdt_true_approx[:, i] = (
            (x[:, ip1] - x[:, im2]) * x[:, im1]
            - x[:, i]
            + F_forcing
        )
    return F.mse_loss(dxdt_pred, dxdt_true_approx)


# ── Hierarchical Symbolic Head ────────────────────────────────────────────────

class HierarchicalSymbolicHead(nn.Module):
    """
    Hierarchical MLP head that approximates governing equations
    in the active subspace using additive univariate and pairwise terms.
    """

    def __init__(self, d, N, hidden=128):
        super().__init__()
        self.d = d
        self.N = N

        # Univariate branch
        self.univariate = nn.Sequential(
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, N)
        )

        # Pairwise interaction branch
        self.pairwise = nn.Sequential(
            nn.Linear(d * d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, N)
        )

        # Output combination
        self.combine = nn.Linear(2 * N, N, bias=True)

        # Initialise combine with small weights for sparse start
        nn.init.xavier_uniform_(self.combine.weight, gain=0.1)
        nn.init.zeros_(self.combine.bias)

        self.l1_weight = 0.001

    def forward(self, x_proj):
        uni  = self.univariate(x_proj)

        batch     = x_proj.shape[0]
        outer     = x_proj.unsqueeze(2) * x_proj.unsqueeze(1)
        outer_flat = outer.view(batch, -1)
        pair      = self.pairwise(outer_flat)

        combined  = torch.cat([uni, pair], dim=-1)
        return self.combine(combined)

    def sparsity_loss(self):
        return self.l1_weight * torch.norm(self.combine.weight, p=1)


# ── Full SymFM Model ──────────────────────────────────────────────────────────

class SymFM(nn.Module):
    def __init__(self, N, d, hidden=128, eta=0.01):
        super().__init__()
        self.N = N
        self.d = d
        self.projection    = ActiveSubspaceProjection(N=N, d=d, eta=eta)
        self.symbolic_head = HierarchicalSymbolicHead(d=d, N=N, hidden=hidden)

    def forward(self, x):
        x_proj    = self.projection(x)
        dxdt_pred = self.symbolic_head(x_proj)
        return dxdt_pred, x_proj

    def compute_loss(self, x, dxdt_true,
                     lambda1=1.0, lambda2=0.5,
                     lambda3=0.1, lambda4=2.0):
        dxdt_pred, x_proj = self.forward(x)

        l_rec     = F.huber_loss(dxdt_pred, dxdt_true, delta=0.5)
        l_sparse  = self.symbolic_head.sparsity_loss()
        l_ortho   = self.projection.orthonormality_loss()
        l_physics = lorenz96_residual(x, dxdt_pred)

        loss = (lambda1 * l_rec
                + lambda2 * l_sparse
                + lambda3 * l_ortho
                + lambda4 * l_physics)

        loss_dict = {
            'total':   float(loss.item()),
            'rec':     float(l_rec.item()),
            'sparse':  float(l_sparse.item()),
            'ortho':   float(l_ortho.item()),
            'physics': float(l_physics.item()),
        }
        return loss, loss_dict


# ── Training ──────────────────────────────────────────────────────────────────

def train_symfm(X_train, dXdt_train, X_val, dXdt_val,
                N, d, n_epochs=2000, lr=5e-4,
                lambda1=1.0, lambda2=0.5,
                lambda3=0.1, lambda4=2.0,
                device='cpu', verbose=True):

    if verbose:
        print(f"\n  Training SymFM: N={N}, d={d}, epochs={n_epochs}, lr={lr}")

    X_tr = torch.tensor(X_train,    dtype=torch.float32).to(device)
    y_tr = torch.tensor(dXdt_train, dtype=torch.float32).to(device)
    X_vl = torch.tensor(X_val,      dtype=torch.float32).to(device)
    y_vl = torch.tensor(dXdt_val,   dtype=torch.float32).to(device)

    model = SymFM(N=N, d=d, hidden=128, eta=0.01).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-5
    )

    history    = {'train_loss': [], 'val_loss': [],
                  'rec': [], 'physics': []}
    best_val   = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        loss, loss_dict = model.compute_loss(
            X_tr, y_tr, lambda1, lambda2, lambda3, lambda4
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history['train_loss'].append(loss_dict['total'])
        history['rec'].append(loss_dict['rec'])
        history['physics'].append(loss_dict['physics'])

        model.eval()
        with torch.no_grad():
            val_loss, _ = model.compute_loss(
                X_vl, y_vl, lambda1, lambda2, lambda3, lambda4
            )
        history['val_loss'].append(float(val_loss.item()))

        if float(val_loss.item()) < best_val:
            best_val   = float(val_loss.item())
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 400 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {loss_dict['total']:.4f} | "
                  f"Val: {val_loss.item():.4f} | "
                  f"Rec: {loss_dict['rec']:.4f} | "
                  f"Physics: {loss_dict['physics']:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_symfm(model, X_test, dXdt_test, device='cpu', tol=0.10):
    model.eval()
    X_t = torch.tensor(X_test,    dtype=torch.float32).to(device)
    y_t = torch.tensor(dXdt_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        dxdt_pred, _ = model.forward(X_t)
        dxdt_pred_np  = dxdt_pred.cpu().numpy()

    num = np.linalg.norm(dXdt_test - dxdt_pred_np)
    den = np.linalg.norm(dXdt_test)
    l2  = float(num / (den + 1e-10))

    recovered  = l2 < tol
    combine_w  = model.symbolic_head.combine.weight.detach().cpu().numpy()
    complexity = int(np.sum(np.abs(combine_w) > 1e-3))

    return {
        'recovered':  recovered,
        'l2':         min(l2, 10.0),
        'complexity': complexity
    }


# ── Main experiment ───────────────────────────────────────────────────────────

def run_symfm_experiment(N, d, data_dir="data", results_dir="results",
                         n_trials=3, n_epochs=2000):

    print(f"\n{'='*60}")
    print(f"SymFM Experiment: N={N}, d={d}")
    print(f"{'='*60}")

    data_path = os.path.join(data_dir, f"lorenz96_N{N}.npz")
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return None

    data     = np.load(data_path)
    X_all    = data['X']
    dXdt_all = data['dXdt']
    n_traj   = X_all.shape[0]
    T        = X_all.shape[1]
    T_fit    = int(0.7 * T)
    T_val    = int(0.15 * T)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Trials: {n_trials}, Epochs: {n_epochs}")

    err_rates    = []
    l2_errors    = []
    complexities = []
    runtimes     = []

    for trial in range(n_trials):
        traj_idx   = trial % n_traj
        X_train    = X_all[traj_idx,   :T_fit,          :]
        dXdt_train = dXdt_all[traj_idx, :T_fit,          :]
        X_val      = X_all[traj_idx,   T_fit:T_fit+T_val, :]
        dXdt_val   = dXdt_all[traj_idx, T_fit:T_fit+T_val, :]
        X_test     = X_all[traj_idx,   T_fit+T_val:,    :]
        dXdt_test  = dXdt_all[traj_idx, T_fit+T_val:,    :]

        print(f"\n  Trial {trial+1}/{n_trials}:")
        start = time.time()

        try:
            model, history = train_symfm(
                X_train, dXdt_train, X_val, dXdt_val,
                N=N, d=d, n_epochs=n_epochs,
                device=device, verbose=True
            )
            runtime = time.time() - start
            metrics = evaluate_symfm(model, X_test, dXdt_test,
                                     device=device, tol=0.10)

            err_rates.append(1.0 if metrics['recovered'] else 0.0)
            l2_errors.append(metrics['l2'])
            complexities.append(metrics['complexity'])
            runtimes.append(runtime)

            status = "RECOVERED" if metrics['recovered'] else "not recovered"
            print(f"    {status} | L2={metrics['l2']:.4f} | "
                  f"complexity={metrics['complexity']} | "
                  f"time={runtime:.1f}s")

            os.makedirs(results_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(results_dir,
                             f"symfm_N{N}_trial{trial}.pt")
            )

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            err_rates.append(0.0)
            l2_errors.append(10.0)
            complexities.append(0)
            runtimes.append(0.0)

    results = {
        "method":          "SymFM",
        "N":               N,
        "d":               d,
        "n_trials":        n_trials,
        "n_epochs":        n_epochs,
        "ERR_mean":        float(np.mean(err_rates) * 100),
        "ERR_std":         float(np.std(err_rates)  * 100),
        "L2_mean":         float(np.mean(l2_errors)),
        "L2_std":          float(np.std(l2_errors)),
        "complexity_mean": float(np.mean(complexities)),
        "runtime_mean_s":  float(np.mean(runtimes)),
        "per_trial": {
            "err_rates":    [float(x) for x in err_rates],
            "l2_errors":    [float(x) for x in l2_errors],
            "complexities": [int(x)   for x in complexities],
            "runtimes":     [float(x) for x in runtimes],
        }
    }

    print(f"\n{'─'*50}")
    print(f"SymFM Results: N={N}")
    print(f"{'─'*50}")
    print(f"  ERR:        {results['ERR_mean']:.1f}%")
    print(f"  L2 Error:   {results['L2_mean']:.4f}")
    print(f"  Complexity: {results['complexity_mean']:.1f} terms")
    print(f"  Runtime:    {results['runtime_mean_s']:.1f} s")

    save_path = os.path.join(results_dir, f"symfm_N{N}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {save_path}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("SymFM Phase 3: Full Model (2000 epochs, physics-informed)")
    print("=" * 60)

    dim_map = {4: 2, 10: 5, 20: 10, 40: 13}

    all_results = {}

    for N, d in dim_map.items():
        results = run_symfm_experiment(
            N=N, d=d,
            data_dir="data",
            results_dir="results",
            n_trials=3,
            n_epochs=2000
        )
        if results:
            all_results[f"N{N}"] = results

    print(f"\n{'='*60}")
    print("SYMFM FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':<6} {'d':<6} {'ERR (%)':<12} "
          f"{'L2 Error':<12} {'Time (s)'}")
    print(f"{'─'*50}")
    for key, r in all_results.items():
        print(f"{r['N']:<6} {r['d']:<6} "
              f"{r['ERR_mean']:<12.1f} "
              f"{r['L2_mean']:<12.4f} "
              f"{r['runtime_mean_s']:.1f}")

    print(f"\nAll results saved in results/ folder.")
    print("Next: run final_comparison.py to compare all methods")
    print("=" * 60)