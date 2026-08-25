"""
Genuine ablation on Lorenz-96 N=4, matching the paper's protocol exactly
(dim_map[4]=2, n_epochs=5000, 3 trials, same data generation call).

Three real, runnable conditions given what the released code actually
implements (see accompanying writeup for why "-Foundation model" and
"-PINN-Obs" are not includable):
  1. Full SymFM         (d=2, lambda4=5.0 -- default)
  2. -Active subspace    (d=N=4, i.e. no dimensionality reduction, lambda4=5.0)
  3. -Physics loss       (d=2, lambda4=0.0)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lorenz96_common import generate_lorenz96_data, train_symfm, evaluate_symfm_model

OUT = os.path.dirname(os.path.abspath(__file__))
N = 4
D_FULL = 2   # dim_map[4] from the original notebook
N_TRIALS = 3
N_EPOCHS = 5000

print(f"Generating Lorenz-96 N={N} data (matches original protocol)...")
X_all, dXdt_all, t_eval = generate_lorenz96_data(N=N, n_trajectories=50, seed=42)
T = X_all.shape[1]
T_fit = int(0.7 * T)
T_val = int(0.15 * T)
n_traj = X_all.shape[0]
print(f"Data ready: {X_all.shape}")

conditions = {
    'full_symfm':          dict(d=D_FULL, lambda4=5.0),
    'no_active_subspace':  dict(d=N,      lambda4=5.0),   # flat, d=N
    'no_physics_loss':     dict(d=D_FULL, lambda4=0.0),
}

all_results = {}
for cond_name, cfg in conditions.items():
    print(f"\n{'='*70}\n{cond_name}  (d={cfg['d']}, lambda4={cfg['lambda4']})\n{'='*70}")
    err_rates, l2_errors, raw_l2s, runtimes = [], [], [], []
    for trial in range(N_TRIALS):
        traj_idx = trial % n_traj
        X_train = X_all[traj_idx, :T_fit, :]
        dXdt_train = dXdt_all[traj_idx, :T_fit, :]
        X_val = X_all[traj_idx, T_fit:T_fit+T_val, :]
        dXdt_val = dXdt_all[traj_idx, T_fit:T_fit+T_val, :]
        X_test = X_all[traj_idx, T_fit+T_val:, :]
        dXdt_test = dXdt_all[traj_idx, T_fit+T_val:, :]

        t0 = time.time()
        model = train_symfm(X_train, dXdt_train, X_val, dXdt_val,
                             N=N, d=cfg['d'], n_epochs=N_EPOCHS, lr=3e-4,
                             lambda4=cfg['lambda4'], device='cpu', seed=trial)
        elapsed = time.time() - t0
        m = evaluate_symfm_model(model, X_test, dXdt_test, device='cpu', tol=0.15)
        err_rates.append(1.0 if m['recovered'] else 0.0)
        l2_errors.append(m['l2'])
        raw_l2s.append(m['raw_l2'])
        runtimes.append(elapsed)
        print(f"  Trial {trial+1}/{N_TRIALS}: recovered={m['recovered']} "
              f"L2={m['l2']:.4f} (raw={m['raw_l2']:.4f})  ({elapsed:.1f}s)")

    result = {
        'condition': cond_name, 'N': N, 'd': cfg['d'], 'lambda4': cfg['lambda4'],
        'n_trials': N_TRIALS, 'n_epochs': N_EPOCHS,
        'ERR_mean': float(np.mean(err_rates) * 100),
        'ERR_std': float(np.std(err_rates) * 100),
        'L2_mean': float(np.mean(l2_errors)),
        'L2_std': float(np.std(l2_errors)),
        'runtime_mean_s': float(np.mean(runtimes)),
        'per_trial': {'err_rates': err_rates, 'l2_errors': l2_errors, 'raw_l2_errors': raw_l2s},
    }
    all_results[cond_name] = result
    print(f"  => ERR_mean={result['ERR_mean']:.1f}%  L2_mean={result['L2_mean']:.4f} (std={result['L2_std']:.4f})")

with open(os.path.join(OUT, 'ablation_N4_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n\n" + "="*70)
print("SUMMARY: Lorenz-96 N=4 ablation")
print("="*70)
print(f"{'Condition':<22}{'ERR%':<10}{'L2_mean':<12}{'L2_std'}")
for name, r in all_results.items():
    print(f"{name:<22}{r['ERR_mean']:<10.1f}{r['L2_mean']:<12.4f}{r['L2_std']:.4f}")
print("\nSaved ablation_N4_results.json")
