<div align="center">

# 🔭 SymFM

### Symbolic Foundation Model for Governing Equation Discovery

*Physics-informed, dimensionality-aware symbolic regression for high-dimensional nonlinear dynamical systems*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](./SymFM_Paper_Final.pdf)
[![Status](https://img.shields.io/badge/status-research%20preview-orange.svg)]()

</div>

---

## Overview

**SymFM** recovers interpretable, closed-form governing equations directly from partial, noisy observations of high-dimensional dynamical systems — without hand-designed function libraries and without architecture redesign per domain.

It combines four ideas that, until now, have lived in separate papers:

| | |
|---|---|
| 🧠 **Foundation model pretraining** | Transformer encoder pretrained on 50,000 synthetic dynamical systems |
| ⚛️ **Physics-informed objectives** | PDE residual losses baked into every training stage |
| 🔤 **Symbolic, interpretable output** | Closed-form equations via hierarchical KAN + PySR extraction |
| 📉 **High-dimensional scalability** | Learned active-subspace projection: `N → d ≪ N` |

---

## Architecture

```
  y(t)              x̂(t)              z              x̃              f̂(x)
partial   ──────►  full state ──────► equation ────► active  ────► symbolic
observ.   Module 1  estimate  Module 2  latent  Module 3 subspace  Module 4  equation
                    (PINN-Obs)         (FM Encoder)     (proj.)   (KAN head)
```

| Module | Role |
|---|---|
| **1 · PINN-Obs** | Reconstructs full state from partial, noisy measurements |
| **2 · FM Encoder** | 4-layer Transformer pretrained to represent *equation structure*, not trajectories |
| **3 · Active Subspace** | Learns a `d × N` projection matrix, collapsing effective dimensionality |
| **4 · KAN Head** | Hierarchical spline regression + PySR/SymPy extraction → readable equation |

---

## Results at a glance

**Lorenz-96** (chaotic, N ∈ {4, 10, 20, 40})

| N | SymFM ℓ₂ | SINDy ℓ₂ | Notes |
|---|---|---|---|
| 4 | 0.100 | 0.000 | Both recover exactly |
| 20 | 0.975 | 1.370 | SINDy begins to collapse |
| 40 | **0.972** | 1.919 | **49% lower error** than SINDy |

**2D Navier–Stokes** (N = 1024) — SymFM is the only evaluated method to recover governing structure at this scale (ℓ₂ = 0.110).

**Spatially heterogeneous SEIRD** (N up to 500) — first systematic high-dimensional evaluation of PINN-Obs; unobserved-compartment RMSE of 0.00353 at N = 500.

> ⚠️ Figures and some result tables in the current paper draft are placeholders / need a final formatting pass — see the paper for full details and caveats.

---

## Benchmarks

- 🌀 **Lorenz-96** — canonical chaotic atmospheric toy model
- 🌊 **2D Navier–Stokes** — incompressible periodic flow, pseudo-spectral solver
- 🦠 **Spatially heterogeneous SEIRD** — epidemiological model with patch coupling

---

## Installation

```bash
git clone https://github.com/Eddiegah/SymFM.git
cd SymFM
pip install -r requirements.txt
```

<details>
<summary>Core dependencies</summary>

| Library | Version |
|---|---|
| PyTorch | 2.x |
| kan | 0.x |
| PySR | 0.19.x |
| SymPy | 1.13.x |
| pysindy | 1.7.x |
| torchdiffeq | 0.2.x |

</details>

---

## Quick start

```python
from symfm import SymFM

model = SymFM(state_dim=40, active_subspace_dim=8)
model.pretrain(corpus="synthetic_dynamics_50k")
model.fit(observations=y, obs_matrix=H)

equation = model.extract_symbolic()
print(equation)
```

---

## Citation

```bibtex
@article{gah2026symfm,
  title   = {Toward Autonomous Scientific Discovery: A Physics-Informed
             Foundation Model with Dimensionality-Aware Symbolic Regression
             for Governing Equation Discovery in High-Dimensional
             Nonlinear Dynamical Systems},
  author  = {Gah, Edmund Eric},
  year    = {2026}
}
```

---

## License

Code released under the [MIT License](LICENSE). Paper text under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless noted otherwise.

---

<div align="center">
<sub>Built with PyTorch · kan · PySR · pysindy · torchdiffeq · Weights & Biases</sub>
</div>
