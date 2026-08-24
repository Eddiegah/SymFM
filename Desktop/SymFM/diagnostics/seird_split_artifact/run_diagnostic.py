"""
Re-run of the SEIRD symbolic-regression step from
notebooks/SymFM_NS_SEIRD_Colab.ipynb, at P=10 (N=50) only (smallest,
fastest case), to (a) confirm the paper's "0% ERR, L2=10.0 (saturated)"
finding reproduces, and (b) diagnose WHY it fails so severely -- L2=10.0
is the eval code's clip cap, meaning the true raw error is far above the
tol=0.25 recovery threshold, not a near-miss.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seird_common import simulate_seird, train_model, eval_model

P = 10
N_s = 5 * P
d_s = 4  # matches paper's seird_d_map = {10: 4, 50: 10, 100: 16}

X_all, dX_all, N_check, params = simulate_seird(P=P, n_steps=300, dt=1.0, n_trajectories=15, seed=42)
assert N_check == N_s

T = X_all.shape[1]
T_fit  = int(0.7 * T)
T_val  = int(0.15 * T)

results = {}

def make_splits(traj_idx=0):
    X_tr  = X_all[traj_idx,   :T_fit,            :]
    y_tr  = dX_all[traj_idx,  :T_fit,            :]
    X_vl  = X_all[traj_idx,   T_fit:T_fit+T_val, :]
    y_vl  = dX_all[traj_idx,  T_fit:T_fit+T_val, :]
    X_tst = X_all[traj_idx,   T_fit+T_val:,      :]
    y_tst = dX_all[traj_idx,  T_fit+T_val:,      :]
    return X_tr, y_tr, X_vl, y_vl, X_tst, y_tst

print("\n=== Data scale sanity check ===")
X_tr, y_tr, X_vl, y_vl, X_tst, y_tst = make_splits(0)
print(f"X range: [{X_tr.min():.6g}, {X_tr.max():.6g}], mean|X|={np.abs(X_tr).mean():.6g}")
print(f"y (dXdt) range: [{y_tr.min():.6g}, {y_tr.max():.6g}], mean|y|={np.abs(y_tr).mean():.6g}")
print(f"per-compartment mean|y|: S={np.abs(y_tr[:,0:P]).mean():.3g} E={np.abs(y_tr[:,P:2*P]).mean():.3g} "
      f"I={np.abs(y_tr[:,2*P:3*P]).mean():.3g} R={np.abs(y_tr[:,3*P:4*P]).mean():.3g} D={np.abs(y_tr[:,4*P:5*P]).mean():.3g}")

print("\n=== Trial 1/1: ORIGINAL config (d=4, 2000 epochs, lr=3e-4, no physics loss) ===")
t0 = time.time()
model, hist = train_model(X_tr, y_tr, X_vl, y_vl, N=N_s, d=d_s, n_epochs=2000,
                           lr=3e-4, physics_fn=None, device='cpu',
                           verbose_every=250, seed=0)
elapsed = time.time() - t0
m = eval_model(model, X_tst, y_tst, device='cpu', tol=0.25)
print(f"  Elapsed: {elapsed:.1f}s")
print(f"  recovered={m['recovered']} l2(clipped)={m['l2']:.4f} raw_l2={m['raw_l2']:.4f}")
print(f"  pred_has_nan={m['pred_has_nan']} pred_scale={m['pred_scale']:.6g} target_scale={m['target_scale']:.6g}")
results['original'] = {**m, 'elapsed_s': elapsed, 'history': hist}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_original.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results_original.json")
