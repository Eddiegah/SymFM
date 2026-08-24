<div align="center">

# 🔭 SymFM

### *Symbolic Foundation Model*

**Physics-Informed, Dimensionality-Aware Symbolic Regression for Governing Equation Discovery in High-Dimensional Nonlinear Dynamical Systems**

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg?style=for-the-badge)]()

<br>

**⚛️ Foundation Model&nbsp;&nbsp;·&nbsp;&nbsp;🧬 Physics-Informed&nbsp;&nbsp;·&nbsp;&nbsp;🔤 Symbolic Output&nbsp;&nbsp;·&nbsp;&nbsp;📈 High-Dimensional&nbsp;&nbsp;·&nbsp;&nbsp;🧩 Zero Domain-Specific Design**

</div>

<br>

---

## ✨ What is SymFM?

> Physical laws have always been *discovered*, not designed. SymFM is a step toward doing that discovery **autonomously, at scale.**

SymFM recovers **interpretable, closed-form governing equations** directly from partial, noisy observations of high-dimensional dynamical systems — no hand-built function library, no domain-specific architecture, no dimensionality ceiling.

It is the first framework to unify five properties that no prior method achieves simultaneously:

<div align="center">

| Property | PINN-Obs | SINDy | KAN | PDE Foundation Models | **SymFM** |
|:---|:---:|:---:|:---:|:---:|:---:|
| Foundation model pretraining | ❌ | ❌ | ❌ | ✅ | **✅** |
| Physics-informed training | ✅ | ❌ | ~ | ~ | **✅** |
| Symbolic interpretable output | ❌ | ✅ | ✅ | ❌ | **✅** |
| Tested at high dimension | ❌ | ❌ | ❌ | ~ | **✅** |
| No domain-specific architecture | ❌ | ❌ | ❌ | ✅ | **✅** |

</div>

---

## 🧠 How it works

```mermaid
flowchart LR
    A["📡 y(t)<br/>partial observations"] -->|"Module 1<br/>PINN-Obs"| B["🧩 x̂(t)<br/>reconstructed state"]
    B -->|"Module 2<br/>Foundation Model Encoder"| C["🌐 z<br/>equation-space latent"]
    C -->|"Module 3<br/>Active Subspace Projection"| D["📉 x̃ ∈ ℝᵈ<br/>d ≪ N"]
    D -->|"Module 4<br/>KAN Symbolic Head"| E["✏️ f̂(x)<br/>closed-form equation"]

    style A fill:#1e293b,stroke:#38bdf8,color:#fff
    style B fill:#1e293b,stroke:#38bdf8,color:#fff
    style C fill:#1e293b,stroke:#a78bfa,color:#fff
    style D fill:#1e293b,stroke:#f472b6,color:#fff
    style E fill:#1e293b,stroke:#34d399,color:#fff
```

<div align="center">

| Module | Component | Role |
|:---:|---|---|
| **1** | 🎛️ PINN-Obs | Reconstructs full state from partial, noisy measurements |
| **2** | 🌐 FM Encoder | 4-layer Transformer, pretrained on 50K synthetic systems to represent *equation structure* |
| **3** | 📉 Active Subspace | Learns a `d × N` projection, collapsing effective dimensionality |
| **4** | ✏️ KAN Head | Hierarchical spline regression → PySR/SymPy → readable equation |

</div>

---

## 📊 Results at a glance

<div align="center">

### Lorenz-96 &nbsp;(chaotic system, N ∈ {4, 10, 20, 40})

| N | SymFM ℓ₂ | SINDy ℓ₂ | Verdict |
|:---:|:---:|:---:|:---|
| 4 | 0.100 | **0.000** | Both recover exactly |
| 20 | **0.975** | 1.370 | SINDy begins to collapse |
| 40 | **0.972** | 1.919 | 🏆 **49% lower error** than SINDy |

### High-Dimensional Benchmarks

| Benchmark | Dimension | SymFM Result | Comparison |
|---|:---:|:---:|---|
| 🌊 2D Navier–Stokes | N = 1,024 | ℓ₂ = 0.110 | **Only method** to recover governing structure at this scale |
| 🦠 Spatially heterogeneous SEIRD (state estimation) | N = 500 | RMSE = 0.00353 | First systematic PINN-Obs eval beyond N = 10 |

</div>

> ⚠️ **Status note (updated):** the paper's Lorenz-96 and Navier-Stokes results (above) have been cross-checked against this repo's actual notebook output and are verified accurate. On SEIRD: end-to-end symbolic equation recovery did not succeed at any tested dimension (0% recovery rate under the paper's strict tolerance; state estimation above succeeds independently of this) — but the *severity* originally reported (a saturated relative-ℓ2 of 10.0 at every N) was a train/test-split artifact, not a purely architectural failure. The benchmark's original chronological split puts the test window in the post-epidemic near-zero-derivative tail of these single-wave trajectories, which destabilizes the relative-ℓ2 metric; see `diagnostics/seird_split_artifact/` for the full diagnosis. Re-run with a representative split and the paper's standard 3-trial protocol (`results/seird_symfm_results.json`), the real result is relative ℓ2 = 0.905±0.006 / 0.902±0.013 / 0.906±0.010 at N=50/250/500 — still short of the 0.25 recovery threshold (so exact recovery genuinely isn't achieved), but flat across a 10× dimension increase and in the same difficulty range as Lorenz-96's hardest cases, rather than a failure that worsens with scale. This is now the paper's reported number, not a diagnostic footnote. The original chronological-split run is kept for the record at `results/seird_symfm_results_chronological_split.json`. Still open: the paper's component-wise ablation study (removing the foundation model / active subspace / PINN-Obs / physics loss one at a time) has no corresponding notebook, script, or logged result in this repo and is marked as future work in the current draft — if you have that run saved elsewhere, add it back under `notebooks/` and `results/`.

---

## 🧪 Benchmark systems

<table>
<tr>
<td width="33%" valign="top">

### 🌀 Lorenz-96
Canonical chaotic atmospheric toy model.
`N ∈ {4, 10, 20, 40}`

</td>
<td width="33%" valign="top">

### 🌊 2D Navier–Stokes
Incompressible periodic flow, custom pseudo-spectral solver.
`N = 1,024`

</td>
<td width="33%" valign="top">

### 🦠 Spatial SEIRD
Epidemiological model with patch-to-patch coupling.
`N ∈ {50, 250, 500}`

</td>
</tr>
</table>

---

## ⚙️ Reproducing the results

This repo is a collection of Colab notebooks and their supporting scripts,
not an installable Python package — there is no `pip install` step and no
`from symfm import SymFM` API. To reproduce a result, open the relevant
notebook in Colab and run it top to bottom on a GPU runtime:

| Notebook | Produces |
|---|---|
| `notebooks/SymFM_Colab.ipynb` | Lorenz-96 main results (Table 3 rows, N=4/10/20/40) |
| `notebooks/SymFM_NS_SEIRD_Colab.ipynb` | Navier-Stokes + SEIRD results (Table 3 NS-32/SEIRD columns, Table 6) |
| `notebooks/SymFM_Sensitivity_Colab_1.ipynb` | Hyperparameter sensitivity grid (Table 5) |

Standalone components used by these notebooks also exist as plain scripts
under `src/` (`symfm_model.py`, `active_subspace.py`, `kan_baseline.py`,
`sindy_baseline.py`, `lorenz96.py`, `phase1_summary.py`,
`scalability_figure.py`), useful for reading the implementation without
launching a notebook.

<details>
<summary><b>📦 Core dependencies</b></summary>
<br>

| Library | Version | Purpose |
|---|:---:|---|
| PyTorch | 2.x | Core deep learning framework |
| kan | 0.x | KAN spline layers |
| PySR | 0.19.x | Symbolic extraction |
| SymPy | 1.13.x | Symbolic simplification |
| pysindy | 1.7.x | SINDy baseline |
| torchdiffeq | 0.2.x | ODE integration |
| Weights & Biases | 0.18.x | Experiment tracking |

</details>

---

## 📁 Repository structure

```
SymFM/
├── src/
│   ├── symfm_model.py           # end-to-end SymFM model
│   ├── active_subspace.py       # Module 3 -- dimensionality reduction
│   ├── kan_baseline.py          # flat KAN baseline (B2)
│   ├── sindy_baseline.py        # SINDy baseline
│   ├── lorenz96.py              # Lorenz-96 data generation
│   ├── phase1_summary.py        # early baseline comparison script
│   └── scalability_figure.py    # Figure 3 generation
├── notebooks/
│   ├── SymFM_Colab.ipynb              # Lorenz-96 (Table 3)
│   ├── SymFM_NS_SEIRD_Colab.ipynb     # Navier-Stokes + SEIRD (Table 3, 6)
│   └── SymFM_Sensitivity_Colab_1.ipynb # Hyperparameter grid (Table 5)
├── data/                        # cached Lorenz-96 .npz trajectories
├── results/                     # per-method-per-N result JSON + checkpoints
├── SymFM_Figures/                # paper figures (PNG)
├── SymFM_Paper_Final.pdf
└── README.md
```

Note: as currently pushed, all of the above lives one directory deeper
than the repo root, under `Desktop/SymFM/` (an artifact of how the repo
was first committed). `git clone` therefore gives you
`SymFM/Desktop/SymFM/...`, not a clean root. Worth moving everything up a
level so `git clone` produces the layout above directly.

---

## 📖 Citation

If you use SymFM in your research, please cite:

```bibtex
@article{gah2026symfm,
  title   = {SymFM: A Physics-Informed Foundation Model with
             Dimensionality-Aware Symbolic Regression for Governing
             Equation Discovery in High-Dimensional Nonlinear
             Dynamical Systems},
  author  = {Gah, Edmund Eric},
  year    = {2026}
}
```

---

## 🪪 License

- **Code** — released under the [MIT License](LICENSE)
- **Paper text** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless noted otherwise

---

<div align="center">

<sub>Built with 🔥 PyTorch · 🧩 kan · ✏️ PySR · 📐 pysindy · 🌀 torchdiffeq · 📊 Weights & Biases</sub>

<br><br>

**⭐ If this project is useful to you, consider starring the repo!**

</div>
