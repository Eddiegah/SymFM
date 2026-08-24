# SEIRD symbolic-regression re-evaluation

The paper reports 0% symbolic Equation Recovery Rate on the SEIRD
benchmark at every tested dimension, with the underlying per-trial log
showing `L2=10.0000` ("not recovered") -- a saturation sentinel, not a
measured value. That's disproportionately worse than every other result
in the paper, so this directory re-runs the evaluation to find out why,
rather than taking an architectural explanation on faith -- and then
runs it properly, matching the paper's own 3-trial protocol.

## Method

`seird_common.py` is the model, training loop, and SEIRD data generator,
extracted structure-for-structure from `notebooks/SymFM_NS_SEIRD_Colab.ipynb`
(cells 4, 10, 12, 14) -- same `SymFM` class, same `train_model`/`eval_model`,
same `simulate_seird`. Run on CPU (no GPU was available for this check).

- `run_diagnostic.py`, `diagnose_metric.py`, `test_random_split.py` --
  the initial diagnostic at P=10 only: reproduce the original
  chronological-split result, decompose the relative-ℓ2 metric by
  evaluation window to isolate the cause, and confirm a representative
  split fixes it.
- `run_full.py` -- extends the chronological-vs-representative-split
  comparison to P=50 (N=250) and P=100 (N=500), single trajectory/seed
  each, as a quick check that the P=10 finding generalises.
- **`run_3trial_protocol.py`** -- the final, statistically proper
  re-evaluation: representative (random) 70/15/15 split, 3 trials per N
  (one trajectory per trial, matching the original notebook's
  trial-indexing convention), mean±std reported the same way Table 3
  reports every other result in the paper. This is what Table 3's SEIRD
  column in the paper is now based on.

## Finding

The SEIRD trajectories are single-wave epidemics that peak around
`t≈90` and decay to a near-quiescent post-epidemic state by `t≈250-300`
(300 timesteps total). The benchmark's **chronological** 70/15/15
train/val/test split puts the test window entirely in that quiescent
tail, where `‖dXdt‖` is orders of magnitude smaller than in the training
window. Relative ℓ2 error divides prediction error by `‖y_test‖`, so it
is numerically unstable exactly when the target norm is tiny.

Full 3-trial, representative-split results at all three tested dimensions:

| N | Chronological-split relative ℓ2 (single trial) | Representative-split relative ℓ2 (mean±std, 3 trials) | ERR |
|---|---|---|---|
| 50  | 86.1 (clipped to the paper's 10.0 sentinel) | 0.905 ± 0.006 | 0% |
| 250 | 240.4 | 0.902 ± 0.013 | 0% |
| 500 | 281.4 | 0.906 ± 0.010 | 0% |

Pooled across all 9 trials: mean = 0.9046, std = 0.0103.

Two things stand out:

1. **The chronological-split failure gets worse with N** (86 → 240 →
   281), consistent with the artifact's cause -- as P grows, the summed
   coupling structure leaves the post-epidemic tail increasingly
   quiescent relative to the training window.
2. **The representative-split result is flat and low-variance** (~0.90-0.91,
   std ≤0.013) across a 10× increase in state dimension. Absolute error
   is comparable across all rows -- the model isn't dramatically better
   or worse at any of them. The relative-ℓ2 metric is what swings wildly
   depending purely on which timesteps land in the denominator.

**This does not overturn the paper's headline result**: 0.90-0.91 is
still above the `tol=0.25` recovery threshold used to compute ERR, so
exact symbolic recovery genuinely was not achieved at any tested N,
across 9 independent trials. But it reframes *why*, with proper
statistical support for the first time: not an order-of-magnitude
architectural failure that gets worse with scale, but a result in the
same difficulty range as Lorenz-96's hardest configurations
(ℓ2=0.97-0.98 at N=20,40) at every tested SEIRD dimension, previously
obscured by an evaluation-protocol artifact.

## What this does not establish

- All training was on CPU; not independently spot-checked against the
  original GPU pipeline for numerical discrepancies (unlikely for this
  architecture, but not directly verified).
- One trajectory per trial (matching the original notebook's
  convention), not multiple trajectories averaged within a trial.
- Whether Lorenz-96/NS-32 have an analogous split issue was not checked
  here (their dynamics are sustained/chaotic rather than a single
  decaying transient, so this specific failure mode is unlikely to
  apply, but that's an inference, not a direct check).

The paper's Table 3 SEIRD column and Discussion section (Section 6) now
report these representative-split, 3-trial numbers directly -- this is
no longer a diagnostic footnote but the benchmark's actual reported
result for SEIRD equation recovery.
