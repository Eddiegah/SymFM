# Lorenz-96 ablation (N=4)

The paper's Table~4 (Section 5.5) previously had no locatable source for
its ablation numbers (see `results/symfm_N*_early_run.json` and the
SEIRD diagnostic for the same pattern elsewhere in this repo). This
directory contains a real, runnable ablation, extracted from
`notebooks/SymFM_Colab.ipynb` (cells 7, 11, 13, 15) -- the exact code
that produced the paper's Lorenz-96 results.

## What's testable, and what isn't

The released code directly supports varying two things within SymFM's
own architecture:
- the active subspace dimension `d` (set `d=N` for "no projection")
- the physics-loss weight `lambda4` in `compute_loss` (set to `0.0` to
  disable it)

It does **not** support a "-foundation model" ablation, because no
foundation-model-pretraining code exists anywhere in this repository
(see the main README's status note) -- there is nothing to switch off.
It also does not meaningfully support a "-PINN-Obs" ablation for
Lorenz-96 specifically, because the Lorenz-96 experiments were run with
full state given directly (no partial-observation reconstruction step),
so removing PINN-Obs is a no-op by construction here (PINN-Obs is only
exercised in the SEIRD benchmark).

## Files

- `lorenz96_common.py` -- model, training loop, and data generator,
  extracted structure-for-structure from the notebook.
- `run_ablation_N4.py` -- runs three conditions (full model, no active
  subspace, no physics loss) at N=4, 3 trials each, 5000 epochs,
  matching the paper's standard protocol exactly (same `dim_map`, same
  data generation call).
- `ablation_N4_results.json` -- raw output.

## Result

| Variant | d | ERR (%) | Relative ℓ2 (mean±std) |
|---|---|---|---|
| Full SymFM | 2 | 66.7 | 0.116 ± 0.039 |
| −Active subspace (d=N) | 4 | 100.0 | 0.004 ± 0.001 |
| −Physics loss (λ4=0) | 2 | 66.7 | 0.131 ± 0.047 |

Removing the physics loss degrades accuracy modestly, as the paper's
design predicts. Removing the active-subspace projection **improves**
accuracy substantially at this dimension. This is not a bug: at N=4 the
projection (d=2) barely compresses anything relative to N=4, so the
ablation mostly measures the cost of a lossy step with no
combinatorial-blowup pressure yet to justify it. The projection's
rationale is a scaling argument (the paper's independently-implemented
flat-KAN baseline collapses at N≥20), and N=4 is precisely the dimension
where that argument doesn't yet apply.

**What this does not establish**: whether the projection's benefit
emerges at N=10, 20, 40, where it is designed to matter. That is the
direct next experiment and was not run here. This is a genuine partial
result, not a complete one -- see the paper's Section 5.5 and Section 6
for how it's written up, including what's still open.
