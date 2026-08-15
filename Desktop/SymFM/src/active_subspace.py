"""
active_subspace.py
------------------
Active Subspace Projection module for SymFM.

Author: Edmund Eric Gah
Project: SymFM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class ActiveSubspaceProjection(nn.Module):
    def __init__(self, N, d, eta=0.01):
        super().__init__()
        self.N = N
        self.d = d
        self.eta = eta
        W_init = torch.randn(d, N)
        Q, _ = torch.linalg.qr(W_init.T)
        W_init = Q[:, :d].T
        self.W = nn.Parameter(W_init)

    def forward(self, x):
        return x @ self.W.T

    def reconstruct(self, x_proj):
        return x_proj @ self.W

    def orthonormality_loss(self):
        WWT = self.W @ self.W.T
        I_d = torch.eye(self.d, device=self.W.device)
        return self.eta * torch.norm(WWT - I_d, p='fro') ** 2

    def get_explained_variance(self, X, dXdt):
        self.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X,    dtype=torch.float32)
            dXdt_t = torch.tensor(dXdt, dtype=torch.float32)
            X_proj = self.forward(X_t)
            X_rec  = self.reconstruct(X_proj)
            var_total = torch.var(dXdt_t).item()
            residual  = dXdt_t - X_rec
            var_resid = torch.var(residual).item()
            explained = max(0.0, 1.0 - var_resid / (var_total + 1e-10))
        return explained


def train_active_subspace(X_train, dXdt_train, N, d=8,
                          n_epochs=500, lr=1e-3, eta=0.01,
                          device='cpu', verbose=True):
    if verbose:
        print(f"\n  Training active subspace: N={N} -> d={d}")

    X_t    = torch.tensor(X_train,    dtype=torch.float32).to(device)
    dXdt_t = torch.tensor(dXdt_train, dtype=torch.float32).to(device)

    proj    = ActiveSubspaceProjection(N=N, d=d, eta=eta).to(device)
    readout = nn.Linear(d, N, bias=True).to(device)

    optimizer = torch.optim.Adam(
        list(proj.parameters()) + list(readout.parameters()), lr=lr
    )

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        X_proj    = proj(X_t)
        dXdt_pred = readout(X_proj)
        loss_proj  = F.mse_loss(dXdt_pred, dXdt_t)
        loss_ortho = proj.orthonormality_loss()
        loss = loss_proj + loss_ortho
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(proj.parameters()) + list(readout.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        losses.append(float(loss.item()))
        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Proj: {loss_proj.item():.4f} | "
                  f"Ortho: {loss_ortho.item():.6f}")

    exp_var = proj.get_explained_variance(X_train, dXdt_train)
    if verbose:
        print(f"  Explained variance: {exp_var*100:.1f}%")

    return proj, losses, readout


def compute_eigenspectrum(X, dXdt, max_d=20):
    T, N   = dXdt.shape
    max_d  = min(max_d, N)
    dXdt_c = dXdt - dXdt.mean(axis=0, keepdims=True)
    C      = (dXdt_c.T @ dXdt_c) / T
    evals  = np.linalg.eigvalsh(C)[::-1]
    evals  = np.abs(evals[:max_d])
    total  = evals.sum() + 1e-10
    explained = np.cumsum(evals) / total
    return evals, explained


if __name__ == "__main__":

    print("=" * 60)
    print("SymFM Phase 3: Active Subspace Projection")
    print("=" * 60)

    results_summary = {}

    for N in [4, 10, 20, 40]:
        data_path = f"data/lorenz96_N{N}.npz"
        if not os.path.exists(data_path):
            print(f"Skipping N={N}: data not found")
            continue

        print(f"\n{'─'*50}")
        print(f"N={N}")
        print(f"{'─'*50}")

        d      = np.load(data_path)
        X_all  = d['X']
        dX_all = d['dXdt']

        X_train    = X_all[0,  :1600, :]
        dXdt_train = dX_all[0, :1600, :]
        X_val      = X_all[1,  1600:, :]
        dXdt_val   = dX_all[1, 1600:, :]

        evals, explained = compute_eigenspectrum(X_train, dXdt_train,
                                                  max_d=min(N, 16))
        print(f"\n  Top eigenvalues and cumulative explained variance:")
        for i, (ev, exp) in enumerate(zip(evals[:6], explained[:6])):
            print(f"    d={i+1}: {ev:.4f}  cumulative={exp*100:.1f}%")

        d_auto = int(np.argmax(explained >= 0.90)) + 1
        d_auto = max(2, min(d_auto, N // 2))
        print(f"\n  Selected d={d_auto} (>= 90% explained variance)")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        proj, losses, readout = train_active_subspace(
            X_train=X_train, dXdt_train=dXdt_train,
            N=N, d=d_auto, n_epochs=500,
            lr=1e-3, eta=0.01, device=device, verbose=True
        )

        exp_train = proj.get_explained_variance(X_train, dXdt_train)
        exp_val   = proj.get_explained_variance(X_val,   dXdt_val)

        print(f"\n  Explained variance train: {exp_train*100:.1f}%")
        print(f"  Explained variance val:   {exp_val*100:.1f}%")
        print(f"  Compression: {N} -> {d_auto} ({N/d_auto:.1f}x)")

        results_summary[f"N{N}"] = {
            "N":           N,
            "d":           d_auto,
            "compression": float(N / d_auto),
            "exp_var_train": float(exp_train),
            "exp_var_val":   float(exp_val),
            "final_loss":    float(losses[-1]),
        }

        os.makedirs("results", exist_ok=True)
        torch.save(
            {'proj_state':    proj.state_dict(),
             'readout_state': readout.state_dict(),
             'N': N, 'd': d_auto, 'losses': losses},
            f"results/active_subspace_N{N}.pt"
        )
        print(f"  Saved: results/active_subspace_N{N}.pt")

    with open("results/active_subspace_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ACTIVE SUBSPACE SUMMARY")
    print(f"{'='*60}")
    print(f"{'N':<6} {'d':<6} {'Compression':<14} "
          f"{'ExpVar Train':<15} {'ExpVar Val'}")
    print(f"{'─'*55}")
    for key, r in results_summary.items():
        print(f"{r['N']:<6} {r['d']:<6} {r['compression']:<14.1f} "
              f"{r['exp_var_train']*100:<15.1f} "
              f"{r['exp_var_val']*100:.1f}%")

    print(f"\nNext step: build symfm_model.py")
    print("=" * 60)