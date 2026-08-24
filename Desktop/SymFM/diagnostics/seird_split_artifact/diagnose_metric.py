"""
Follow-up diagnostic: is the SEIRD "0% recovery" a real model failure, or an
artifact of (a) the chronological 70/15/15 train/val/test split landing the
test window in a post-epidemic near-zero-derivative regime, which blows up a
*relative* L2 metric even for a small absolute error?
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
T_fit  = int(0.7 * T)
T_val  = int(0.15 * T)

traj_idx = 0
X_tr  = X_all[traj_idx,   :T_fit,            :]
y_tr  = dX_all[traj_idx,  :T_fit,            :]
X_vl  = X_all[traj_idx,   T_fit:T_fit+T_val, :]
y_vl  = dX_all[traj_idx,  T_fit:T_fit+T_val, :]
X_tst = X_all[traj_idx,   T_fit+T_val:,      :]
y_tst = dX_all[traj_idx,  T_fit+T_val:,      :]

print("=== Per-segment |dXdt| norms (confirms whether test window is quiescent) ===")
print(f"train (t=0..{T_fit}):        mean|y|={np.abs(y_tr).mean():.6g}  ||y||={np.linalg.norm(y_tr):.6g}")
print(f"val   (t={T_fit}..{T_fit+T_val}):   mean|y|={np.abs(y_vl).mean():.6g}  ||y||={np.linalg.norm(y_vl):.6g}")
print(f"test  (t={T_fit+T_val}..{T}):  mean|y|={np.abs(y_tst).mean():.6g}  ||y||={np.linalg.norm(y_tst):.6g}")

print("\n=== I compartment trajectory over time (epidemic curve shape) ===")
I_traj = X_all[traj_idx, :, 2*P:3*P].mean(axis=1)  # mean across patches
for chunk_start in range(0, T, 30):
    chunk = I_traj[chunk_start:chunk_start+30]
    print(f"  t={chunk_start:>3}-{chunk_start+30:>3}: mean I={chunk.mean():.6g}")

print("\n=== Re-training and evaluating on train-window vs test-window ===")
model, hist = train_model(X_tr, y_tr, X_vl, y_vl, N=N_s, d=d_s, n_epochs=2000,
                           lr=3e-4, physics_fn=None, device='cpu', seed=0)

m_test = eval_model(model, X_tst, y_tst, device='cpu', tol=0.25)
m_train_selfeval = eval_model(model, X_tr, y_tr, device='cpu', tol=0.25)

# absolute-error metrics, scale-independent of ||y||
model.eval()
with torch.no_grad():
    pred_test, _ = model(torch.tensor(X_tst, dtype=torch.float32))
    pred_train, _ = model(torch.tensor(X_tr, dtype=torch.float32))
mae_test = float(np.abs(pred_test.numpy() - y_tst).mean())
mae_train = float(np.abs(pred_train.numpy() - y_tr).mean())
rmse_test = float(np.sqrt(((pred_test.numpy() - y_tst)**2).mean()))
rmse_train = float(np.sqrt(((pred_train.numpy() - y_tr)**2).mean()))

print(f"\nOn TEST window:  relative_l2(raw)={m_test['raw_l2']:.4f}  MAE={mae_test:.6g}  RMSE={rmse_test:.6g}  ||y_test||={np.linalg.norm(y_tst):.6g}")
print(f"On TRAIN window: relative_l2(raw)={m_train_selfeval['raw_l2']:.4f}  MAE={mae_train:.6g}  RMSE={rmse_train:.6g}  ||y_train||={np.linalg.norm(y_tr):.6g}")

print("\n=== Trying eval on a MIDDLE window instead of the tail (same model, no retraining) ===")
mid_start = T // 3
mid_end = 2 * T // 3
X_mid = X_all[traj_idx, mid_start:mid_end, :]
y_mid = dX_all[traj_idx, mid_start:mid_end, :]
m_mid = eval_model(model, X_mid, y_mid, device='cpu', tol=0.25)
print(f"Middle-window (t={mid_start}-{mid_end}) eval: relative_l2(raw)={m_mid['raw_l2']:.4f} recovered={m_mid['recovered']} ||y_mid||={np.linalg.norm(y_mid):.6g}")

results = {
    'test_relative_l2': m_test['raw_l2'], 'test_mae': mae_test, 'test_rmse': rmse_test, 'test_target_norm': float(np.linalg.norm(y_tst)),
    'train_relative_l2': m_train_selfeval['raw_l2'], 'train_mae': mae_train, 'train_rmse': rmse_train, 'train_target_norm': float(np.linalg.norm(y_tr)),
    'mid_relative_l2': m_mid['raw_l2'], 'mid_recovered': m_mid['recovered'], 'mid_target_norm': float(np.linalg.norm(y_mid)),
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_metric_diagnosis.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved results_metric_diagnosis.json")
