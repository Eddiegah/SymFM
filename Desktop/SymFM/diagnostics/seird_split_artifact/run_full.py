"""
Extends the split-artifact diagnostic to P=50 (N=250) and P=100 (N=500),
matching the paper's seird_d_map = {10: 4, 50: 10, 100: 16}. Only
generates 1 trajectory per P (not the original 15) since the diagnostic
only ever uses traj_idx=0 -- saves solve_ivp cost, doesn't change the
per-trajectory dynamics being tested.
"""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\gahed\AppData\Local\Temp\claude\C--Projects-SignBridge\156f2dce-8240-4a38-b310-1446d952ba29\scratchpad\seird_rerun')
from seird_common import simulate_seird, train_model, eval_model

OUT = r'C:\Users\gahed\AppData\Local\Temp\claude\C--Projects-SignBridge\156f2dce-8240-4a38-b310-1446d952ba29\scratchpad\seird_rerun'
seird_d_map = {10: 4, 50: 10, 100: 16}

def run_for_P(P, n_epochs=2000):
    N_s = 5 * P
    d_s = seird_d_map[P]
    print(f"\n{'='*70}\nP={P}  N={N_s}  d={d_s}\n{'='*70}")

    t_sim0 = time.time()
    X_all, dX_all, N_check, params = simulate_seird(P=P, n_steps=300, dt=1.0, n_trajectories=1, seed=42)
    print(f"  data gen: {time.time()-t_sim0:.1f}s")
    assert N_check == N_s

    T = X_all.shape[1]
    T_fit  = int(0.7 * T)
    T_val  = int(0.15 * T)
    X_full, y_full = X_all[0], dX_all[0]

    # --- chronological split (original benchmark protocol) ---
    X_tr, y_tr = X_full[:T_fit], y_full[:T_fit]
    X_vl, y_vl = X_full[T_fit:T_fit+T_val], y_full[T_fit:T_fit+T_val]
    X_tst, y_tst = X_full[T_fit+T_val:], y_full[T_fit+T_val:]

    print(f"  chronological split norms: ||y_tr||={np.linalg.norm(y_tr):.4g} "
          f"||y_vl||={np.linalg.norm(y_vl):.4g} ||y_tst||={np.linalg.norm(y_tst):.4g}")

    t0 = time.time()
    model_chrono, _ = train_model(X_tr, y_tr, X_vl, y_vl, N=N_s, d=d_s, n_epochs=n_epochs,
                                   lr=3e-4, physics_fn=None, device='cpu', seed=0)
    elapsed_chrono = time.time() - t0
    m_chrono = eval_model(model_chrono, X_tst, y_tst, device='cpu', tol=0.25)
    print(f"  chronological-split train: {elapsed_chrono:.1f}s -> "
          f"recovered={m_chrono['recovered']} relative_l2={m_chrono['raw_l2']:.4f} "
          f"(clipped={m_chrono['l2']:.4f})")

    # --- random (representative) split ---
    rng = np.random.RandomState(0)
    perm = rng.permutation(T)
    n_tr = int(0.7 * T); n_vl = int(0.15 * T)
    idx_tr, idx_vl, idx_tst = perm[:n_tr], perm[n_tr:n_tr+n_vl], perm[n_tr+n_vl:]
    Xr_tr, yr_tr = X_full[idx_tr], y_full[idx_tr]
    Xr_vl, yr_vl = X_full[idx_vl], y_full[idx_vl]
    Xr_tst, yr_tst = X_full[idx_tst], y_full[idx_tst]

    print(f"  random split norms: ||y_tr||={np.linalg.norm(yr_tr):.4g} "
          f"||y_vl||={np.linalg.norm(yr_vl):.4g} ||y_tst||={np.linalg.norm(yr_tst):.4g}")

    t0 = time.time()
    model_rand, _ = train_model(Xr_tr, yr_tr, Xr_vl, yr_vl, N=N_s, d=d_s, n_epochs=n_epochs,
                                 lr=3e-4, physics_fn=None, device='cpu', seed=0)
    elapsed_rand = time.time() - t0
    m_rand = eval_model(model_rand, Xr_tst, yr_tst, device='cpu', tol=0.25)
    print(f"  random-split train: {elapsed_rand:.1f}s -> "
          f"recovered={m_rand['recovered']} relative_l2={m_rand['raw_l2']:.4f}")

    return {
        'P': P, 'N': N_s, 'd': d_s,
        'chronological': {'relative_l2': m_chrono['raw_l2'], 'clipped_l2': m_chrono['l2'],
                           'recovered': m_chrono['recovered'], 'elapsed_s': elapsed_chrono,
                           'y_train_norm': float(np.linalg.norm(y_tr)), 'y_test_norm': float(np.linalg.norm(y_tst))},
        'random_split': {'relative_l2': m_rand['raw_l2'], 'recovered': m_rand['recovered'],
                          'elapsed_s': elapsed_rand,
                          'y_train_norm': float(np.linalg.norm(yr_tr)), 'y_test_norm': float(np.linalg.norm(yr_tst))},
    }

if __name__ == '__main__':
    import sys as _sys
    P_list = [int(x) for x in _sys.argv[1:]] if len(_sys.argv) > 1 else [50, 100]
    all_results = {}
    for P in P_list:
        all_results[f'P{P}'] = run_for_P(P)
        with open(f'{OUT}/results_full_N{5*P}.json', 'w') as f:
            json.dump(all_results[f'P{P}'], f, indent=2)
        print(f"  saved results_full_N{5*P}.json")
    print("\n\nDone:", json.dumps(all_results, indent=2))
