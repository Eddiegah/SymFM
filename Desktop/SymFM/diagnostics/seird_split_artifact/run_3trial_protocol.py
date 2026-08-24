"""
Full 3-trial protocol for SEIRD symbolic regression using a
REPRESENTATIVE (random) 70/15/15 split instead of the original
benchmark's chronological split -- matching the trial-count and
mean/std reporting style used elsewhere in the paper's Table 3
(n_trials=3, one trajectory per trial, ERR_mean/ERR_std/L2_mean/L2_std).

This supersedes the single-trajectory/single-seed diagnostic in
run_full.py: same architecture, same hyperparameters, same d per N
(seird_d_map), only difference from the original benchmark protocol is
using a random split instead of a chronological one, per the finding in
results_full_N*.json / diagnose_metric.py.
"""
import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seird_common import simulate_seird, train_model, eval_model

OUT = os.path.dirname(os.path.abspath(__file__))
seird_d_map = {10: 4, 50: 10, 100: 16}
N_TRIALS = 3

def run_for_P(P, n_epochs=2000):
    N_s = 5 * P
    d_s = seird_d_map[P]
    print(f"\n{'='*70}\nP={P}  N={N_s}  d={d_s}  ({N_TRIALS} trials, representative split)\n{'='*70}")

    t_sim0 = time.time()
    X_all, dX_all, N_check, params = simulate_seird(P=P, n_steps=300, dt=1.0,
                                                       n_trajectories=N_TRIALS, seed=42)
    print(f"  data gen ({N_TRIALS} trajectories): {time.time()-t_sim0:.1f}s")
    assert N_check == N_s
    T = X_all.shape[1]

    err_list, l2_list, elapsed_list = [], [], []
    for trial in range(N_TRIALS):
        X_full, y_full = X_all[trial], dX_all[trial]

        rng = np.random.RandomState(trial)
        perm = rng.permutation(T)
        n_tr = int(0.7 * T); n_vl = int(0.15 * T)
        idx_tr, idx_vl, idx_tst = perm[:n_tr], perm[n_tr:n_tr+n_vl], perm[n_tr+n_vl:]
        X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
        X_vl, y_vl = X_full[idx_vl], y_full[idx_vl]
        X_tst, y_tst = X_full[idx_tst], y_full[idx_tst]

        t0 = time.time()
        model, _ = train_model(X_tr, y_tr, X_vl, y_vl, N=N_s, d=d_s, n_epochs=n_epochs,
                                lr=3e-4, physics_fn=None, device='cpu', seed=trial)
        elapsed = time.time() - t0
        m = eval_model(model, X_tst, y_tst, device='cpu', tol=0.25)
        err_list.append(1.0 if m['recovered'] else 0.0)
        l2_list.append(m['raw_l2'])
        elapsed_list.append(elapsed)
        print(f"  Trial {trial+1}/{N_TRIALS}: recovered={m['recovered']} "
              f"relative_l2={m['raw_l2']:.4f}  ({elapsed:.1f}s)")

    result = {
        'method': 'SymFM (representative split)',
        'N': N_s, 'P': P, 'd': d_s, 'n_trials': N_TRIALS, 'n_epochs': n_epochs,
        'ERR_mean': float(np.mean(err_list) * 100),
        'ERR_std': float(np.std(err_list) * 100),
        'L2_mean': float(np.mean(l2_list)),
        'L2_std': float(np.std(l2_list)),
        'runtime_mean_s': float(np.mean(elapsed_list)),
        'per_trial': {'err_rates': err_list, 'l2_errors': l2_list, 'runtimes': elapsed_list},
    }
    print(f"  => ERR_mean={result['ERR_mean']:.1f}% (std={result['ERR_std']:.1f}) "
          f"L2_mean={result['L2_mean']:.4f} (std={result['L2_std']:.4f})")
    return result

if __name__ == '__main__':
    P_list = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [10, 50, 100]
    all_results = {}
    for P in P_list:
        res = run_for_P(P)
        all_results[f'P{P}'] = res
        with open(f'{OUT}/results_3trial_N{5*P}.json', 'w') as f:
            json.dump(res, f, indent=2)
        print(f"  saved results_3trial_N{5*P}.json")

    print("\n\n" + "="*70)
    print("SUMMARY: SEIRD representative-split, 3-trial protocol")
    print("="*70)
    print(f"{'N':<8}{'ERR_mean':<12}{'L2_mean':<12}{'L2_std'}")
    for P in P_list:
        r = all_results[f'P{P}']
        print(f"{r['N']:<8}{r['ERR_mean']:<12.1f}{r['L2_mean']:<12.4f}{r['L2_std']:.4f}")
