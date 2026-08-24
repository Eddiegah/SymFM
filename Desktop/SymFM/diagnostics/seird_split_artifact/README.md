# SEIRD symbolic-regression re-evaluation

The paper reports 0% symbolic Equation Recovery Rate on the SEIRD
benchmark at every tested dimension, with the underlying per-trial log
showing `L2=10.0000` ("not recovered") -- a saturation sentinel, not a
measured value. That's disproportionately worse than every other result
in the paper, so this directory re-runs the evaluation to find out why,
rather than taking an architectural explanation on faith.

## Method

`seird_common.py` is the model, training loop, and SEIRD data generator,
extracted structure-for-structure from `notebooks/SymFM_NS_SEIRD_Colab.ipynb`
(cells 4, 10, 12, 14) -- same `SymFM` class, same `train_model`/`eval_model`,
same `simulate_seird`. Re-run on CPU (no GPU was available for this
check), as a diagnostic rather than a full reproduction of the paper's
3-trial protocol.

- `run_diagnostic.py` -- reproduces the original chronological-split
  result at P=10 and prints per-compartment target scale.
- `diagnose_metric.py` -- breaks the relative-ℓ2 metric down by
  train/val/test window and by absolute error (MAE/RMSE) to isolate the
  cause of the blow-up, at P=10.
- `test_random_split.py` -- re-trains and re-evaluates P=10 using a
  random (representative) 70/15/15 split instead of the original
  chronological one, as the direct test of the split-artifact hypothesis.
- `run_full.py` -- extends the chronological-vs-representative-split
  comparison to P=50 (N=250) and P=100 (N=500), matching the paper's
  `seird_d_map = {10: 4, 50: 10, 100: 16}`. Only generates 1 trajectory
  per P (not the original 15), since the diagnostic only ever uses
  `traj_idx=0` -- saves `solve_ivp` cost without changing the dynamics
  being tested.

## Finding

The SEIRD trajectories are single-wave epidemics that peak around
`t≈90` and decay to a near-quiescent post-epidemic state by `t≈250-300`
(300 timesteps total). The benchmark's **chronological** 70/15/15
train/val/test split puts the test window entirely in that quiescent
tail, where `‖dXdt‖` is orders of magnitude smaller than in the training
window. Relative ℓ2 error divides prediction error by `‖y_test‖`, so it
is numerically unstable exactly when the target norm is tiny.

Checked at all three tested dimensions:

| N | Chronological-split relative ℓ2 | Representative-split relative ℓ2 |
|---|---|---|
| 50  | 86.1 (clipped to the paper's 10.0 sentinel) | 0.910 |
| 250 | 240.4 | 0.913 |
| 500 | 281.4 | 0.915 |

Two things stand out:

1. **The chronological-split failure gets worse with N** (86 → 240 →
   281), consistent with the artifact's cause -- as P grows, the summed
   coupling structure leaves the post-epidemic tail increasingly
   quiescent relative to the training window.
2. **The representative-split result is essentially flat** (~0.91-0.92)
   across a 10× increase in state dimension. Absolute error is
   comparable across all rows -- the model isn't dramatically better or
   worse at any of them. The relative-ℓ2 metric is what swings wildly
   depending purely on which timesteps land in the denominator.

**This does not overturn the paper's headline result**: 0.91-0.92 is
still above the `tol=0.25` recovery threshold used to compute ERR, so
exact symbolic recovery genuinely was not achieved at any tested N. But
it reframes *why*, and reveals something positive: not an
order-of-magnitude architectural failure that gets worse with scale, but
a result in the same difficulty range as Lorenz-96's hardest
configurations (ℓ2=0.97-0.98 at N=20,40) at every tested SEIRD
dimension, previously obscured by an evaluation-protocol artifact.

## What this does not establish

- Each N was checked with a single trajectory and a single random-split
  seed, not the paper's 3-trial protocol with reported mean/std.
- Whether Lorenz-96/NS-32 have an analogous issue was not checked here
  (their dynamics are sustained/chaotic rather than a single decaying
  transient, so this specific failure mode is unlikely to apply, but
  that's an inference, not a direct check).

See `main.tex`'s Discussion section (and the `TODO(author)` comment
right after this paragraph) in the paper repo for how this is written up,
and what's still needed before treating it as final benchmark
methodology rather than a diagnostic: re-running with the full 3-trial
protocol and deciding whether to promote these representative-split
numbers into Table 3's SEIRD column itself.
