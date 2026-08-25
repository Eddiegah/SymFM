"""
lorenz96.py
-----------
Generates simulation data for the Lorenz-96 dynamical system.

Lorenz-96 governing equation:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    for i = 1, ..., N (indices modulo N)

This is the ground truth equation we want SymFM to recover.
Phase 1: Baseline data generation for N=4 and N=10.

Author: Edmund Eric Gah
Project: SymFM
"""

import numpy as np
from scipy.integrate import solve_ivp
import os

# ── Lorenz-96 vector field ────────────────────────────────────────────────────

def lorenz96(t, x, F=8.0):
    """
    Computes the time derivative of the Lorenz-96 system.

    Parameters
    ----------
    t : float
        Current time (not used directly, system is autonomous).
    x : np.ndarray, shape (N,)
        Current state vector.
    F : float
        Forcing constant. F=8 produces chaotic behaviour for N >= 4.

    Returns
    -------
    dxdt : np.ndarray, shape (N,)
        Time derivative at current state.
    """
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] \
                  - x[i] + F
    return dxdt


# ── Data generation ───────────────────────────────────────────────────────────

def generate_lorenz96_data(
    N,
    F=8.0,
    t_start=0.0,
    t_end=20.0,
    dt=0.01,
    n_trajectories=50,
    noise_level=0.01,
    seed=42,
    save_dir="data"
):
    """
    Generates multiple Lorenz-96 trajectories from random initial conditions.

    Parameters
    ----------
    N : int
        State dimension (number of variables).
    F : float
        Forcing constant (default 8.0 for chaotic regime).
    t_start : float
        Start time.
    t_end : float
        End time.
    dt : float
        Time step for output.
    n_trajectories : int
        Number of independent trajectories to generate.
    noise_level : float
        Standard deviation of Gaussian noise added to observations.
    seed : int
        Random seed for reproducibility.
    save_dir : str
        Directory to save the generated data.

    Returns
    -------
    dict with keys:
        'X'       : np.ndarray, shape (n_trajectories, T, N) - clean states
        'X_noisy' : np.ndarray, shape (n_trajectories, T, N) - noisy observations
        'dXdt'    : np.ndarray, shape (n_trajectories, T, N) - time derivatives
        't'       : np.ndarray, shape (T,) - time points
        'N'       : int - state dimension
        'F'       : float - forcing constant
    """
    np.random.seed(seed)

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end, dt)
    T = len(t_eval)

    X_all = np.zeros((n_trajectories, T, N))
    dXdt_all = np.zeros((n_trajectories, T, N))

    print(f"\nGenerating Lorenz-96 data: N={N}, F={F}")
    print(f"Trajectories: {n_trajectories}, Time steps: {T}, dt={dt}")
    print(f"Noise level: {noise_level}")
    print("-" * 50)

    for traj_idx in range(n_trajectories):
        # Random initial condition near the attractor
        x0 = F * np.ones(N)
        x0[0] += 0.01 * np.random.randn()  # small perturbation
        x0 += 0.5 * np.random.randn(N)     # additional randomness

        # Integrate the ODE
        sol = solve_ivp(
            fun=lambda t, x: lorenz96(t, x, F=F),
            t_span=t_span,
            y0=x0,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )

        if not sol.success:
            print(f"  Warning: trajectory {traj_idx} integration failed: {sol.message}")
            continue

        X_traj = sol.y.T  # shape (T, N)

        # Compute time derivatives from the vector field (ground truth)
        dXdt_traj = np.array([
            lorenz96(t_eval[k], X_traj[k], F=F)
            for k in range(T)
        ])

        X_all[traj_idx] = X_traj
        dXdt_all[traj_idx] = dXdt_traj

        if (traj_idx + 1) % 10 == 0:
            print(f"  Generated {traj_idx + 1}/{n_trajectories} trajectories")

    # Add observation noise
    X_noisy = X_all + noise_level * np.random.randn(*X_all.shape)

    # Save data
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"lorenz96_N{N}.npz")
    np.savez(
        save_path,
        X=X_all,
        X_noisy=X_noisy,
        dXdt=dXdt_all,
        t=t_eval,
        N=N,
        F=F
    )
    print(f"\nData saved to: {save_path}")
    print(f"Clean data shape:    {X_all.shape}   (trajectories x timesteps x state_dim)")
    print(f"Noisy data shape:    {X_noisy.shape}")
    print(f"Derivatives shape:   {dXdt_all.shape}")

    return {
        'X': X_all,
        'X_noisy': X_noisy,
        'dXdt': dXdt_all,
        't': t_eval,
        'N': N,
        'F': F
    }


# ── Train / val / test split ──────────────────────────────────────────────────

def split_data(data_dict, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Splits trajectories into train, validation, and test sets.

    Parameters
    ----------
    data_dict : dict
        Output from generate_lorenz96_data.
    train_frac : float
        Fraction of trajectories for training.
    val_frac : float
        Fraction for validation (remainder goes to test).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing
    sub-dicts with 'X', 'X_noisy', 'dXdt', 't'.
    """
    np.random.seed(seed)
    n = data_dict['X'].shape[0]
    indices = np.random.permutation(n)

    n_train = int(train_frac * n)
    n_val   = int(val_frac * n)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def subset(idx):
        return {
            'X':       data_dict['X'][idx],
            'X_noisy': data_dict['X_noisy'][idx],
            'dXdt':    data_dict['dXdt'][idx],
            't':       data_dict['t'],
            'N':       data_dict['N'],
            'F':       data_dict['F']
        }

    splits = {
        'train': subset(train_idx),
        'val':   subset(val_idx),
        'test':  subset(test_idx)
    }

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} trajectories")
    print(f"  Val:   {len(val_idx)} trajectories")
    print(f"  Test:  {len(test_idx)} trajectories")

    return splits


# ── Quick sanity check ────────────────────────────────────────────────────────

def verify_ground_truth(data_dict, traj_idx=0, t_idx=100):
    """
    Verifies that the stored derivatives match the vector field.
    If this passes, the data generation is correct.
    """
    x = data_dict['X'][traj_idx, t_idx]
    dxdt_stored = data_dict['dXdt'][traj_idx, t_idx]
    dxdt_computed = lorenz96(0.0, x, F=data_dict['F'])
    error = np.max(np.abs(dxdt_stored - dxdt_computed))
    print(f"\nGround truth verification:")
    print(f"  Max derivative error: {error:.2e}")
    if error < 1e-6:
        print(f"  PASSED -- derivatives are consistent with the vector field")
    else:
        print(f"  WARNING -- derivatives do not match, check data generation")
    return error


# ── Main: generate data for both N=4 and N=10 ─────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("SymFM Phase 1: Lorenz-96 Data Generation")
    print("=" * 60)

    for N in [4, 10, 20, 40]:
        data = generate_lorenz96_data(
            N=N,
            F=8.0,
            t_start=0.0,
            t_end=20.0,
            dt=0.01,
            n_trajectories=50,
            noise_level=0.01,
            seed=42,
            save_dir="data"
        )
        verify_ground_truth(data)
        splits = split_data(data)
        print()

    print("=" * 60)
    print("Data generation complete.")
    print("Files saved in the data/ folder:")
    print("  data/lorenz96_N4.npz")
    print("  data/lorenz96_N10.npz")
    print("  data/lorenz96_N20.npz")
    print("  data/lorenz96_N40.npz")
    print("=" * 60)