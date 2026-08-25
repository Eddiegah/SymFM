"""
Same SEIRD P=10 setup, but with a RANDOM (stratified across the epidemic
curve) train/val/test split instead of the original chronological
first-70%/next-15%/last-15% split. Tests whether the "0% ERR" result was
mainly a split/metric artifact (test window landing in the post-epidemic
near-zero-derivative regime) rather than a fundamental architecture limit.
"""
import sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seird_common import simulate_seird, train_model, eval_model

P = 10
N_s = 5 * P
d_s = 4

X_all, dX_all, N_check, params = simulate_seird(P=P, n_steps=300, dt=1.0, n_trajectories=15, seed=42)
T = X_all.shape[1]
traj_idx = 0
X_full = X_all[traj_idx]
y_full = dX_all[traj_idx]

rng = np.random.RandomState(0)
perm = rng.permutation(T)
n_tr = int(0.7 * T); n_vl = int(0.15 * T)
idx_tr, idx_vl, idx_tst = perm[:n_tr], perm[n_tr:n_tr+n_vl], perm[n_tr+n_vl:]

X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
X_vl, y_vl = X_full[idx_vl], y_full[idx_vl]
X_tst, y_tst = X_full[idx_tst], y_full[idx_tst]

print(f"Random split norms: ||y_tr||={np.linalg.norm(y_tr):.4g} ||y_vl||={np.linalg.norm(y_vl):.4g} ||y_tst||={np.linalg.norm(y_tst):.4g}")
print("(should all be comparable now, unlike the chronological split's 340x train/test gap)")

model, hist = train_model(X_tr, y_tr, X_vl, y_vl, N=N_s, d=d_s, n_epochs=2000,
                           lr=3e-4, physics_fn=None, device='cpu', seed=0)
m = eval_model(model, X_tst, y_tst, device='cpu', tol=0.25)
print(f"\nRandom-split test result: recovered={m['recovered']} relative_l2={m['raw_l2']:.4f} (tol=0.25)")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_random_split.json'), 'w') as f:
    json.dump({'relative_l2': m['raw_l2'], 'recovered': m['recovered'],
               'y_tr_norm': float(np.linalg.norm(y_tr)), 'y_tst_norm': float(np.linalg.norm(y_tst))}, f, indent=2)
print("Saved results_random_split.json")
