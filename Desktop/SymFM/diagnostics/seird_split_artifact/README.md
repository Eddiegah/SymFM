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
same `simulate_seird`. Re-run at `P=10` (`N=50`) only, on CPU (no GPU was
available for this check), as a diagnostic rather than a full reproduction
of the paper's 3-trial protocol.

- `run_diagnostic.py` -- reproduces the original chronological-split
  result and prints per-compartment target scale.
- `diagnose_metric.py` -- breaks the relative-ℓ2 metric down by
  train/val/test window and by absolute error (MAE/RMSE) to isolate the
  cause of the blow-up.
- `test_random_split.py` -- re-trains and re-evaluates using a random
  (representative) 70/15/15 split instead of the original chronological
  one, as the direct test of the split-artifact hypothesis.

## Finding

The SEIRD trajectories are single-wave epidemics that peak around
`t≈90` and decay to a near-quiescent post-epidemic state by `t≈250-300`
(300 timesteps total). The benchmark's **chronological** 70/15/15
train/val/test split puts the test window entirely in that quiescent
tail, where `‖dXdt‖` is ~340× smaller than in the training window
(`‖y_test‖=1.7e-3` vs. `‖y_train‖=0.575`). Relative ℓ2 error divides
prediction error by `‖y_test‖`, so it is numerically unstable exactly
when the target norm is tiny.

Confirmed directly:

| Evaluation | relative ℓ2 | absolute MAE |
|---|---|---|
| Original chronological-split test window | **86.1** (clipped to 10.0 in the paper's sentinel) | 1.85e-3 |
| Same model, train window | 0.84 | 2.66e-3 |
| Same model, middle window (t=100-200, near epidemic peak) | 0.83 | -- |
| **Retrained on a random 70/15/15 split** | **0.91** | -- |

The absolute error is comparable across all four rows -- the model isn't
dramatically better or worse at any of them. The relative-ℓ2 metric is
what swings from 86.1 to 0.91 depending purely on which timesteps land
in the denominator.

**This does not overturn the paper's headline result**: 0.91 is still
above the `tol=0.25` recovery threshold used to compute ERR, so exact
symbolic recovery genuinely was not achieved. But it reframes *why*:
not an order-of-magnitude architectural failure, but a result in the
same difficulty range as Lorenz-96's hardest configurations (ℓ2=0.97-0.98
at N=20,40), previously obscured by an evaluation-protocol artifact.

## What this does not establish

- Only checked at `P=10` (`N=50`); `P=50,100` (`N=250,500`) were not
  re-run.
- Only one random-split seed; the paper's other benchmarks use 3 trials.
- Whether Lorenz-96/NS-32 have an analogous issue was not checked here
  (their dynamics are sustained/chaotic rather than a single decaying
  transient, so this specific failure mode is unlikely to apply, but
  that's an inference, not a direct check).

See `main.tex`'s Discussion section (and the `TODO(author)` comment
right after this paragraph) in the paper repo for how this is written up,
and what's still needed before treating it as final: re-running at
N=250/500 with the full 3-trial protocol and a representative split as
the actual benchmark methodology, not just this diagnostic.
