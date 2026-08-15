<div align="center">

# 🔭 SymFM

### *Symbolic Foundation Model*

**Physics-Informed, Dimensionality-Aware Symbolic Regression for Governing Equation Discovery in High-Dimensional Nonlinear Dynamical Systems**

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg?style=for-the-badge&logo=readthedocs&logoColor=white)](./SymFM_Paper_Final.pdf)
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
| 🦠 Spatially heterogeneous SEIRD | N = 500 | RMSE = 0.00353 | First systematic PINN-Obs eval beyond N = 10 |

</div>

> ⚠️ **Honest note:** figures and a few result tables in the current paper draft are still placeholders and need a final formatting/verification pass before wider circulation — see the [paper](./SymFM_Paper_Final.pdf) for full methodology and caveats.

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

## ⚙️ Installation

```bash
git clone https://github.com/Eddiegah/SymFM.git
cd SymFM
pip install -r requirements.txt
```

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

## 🚀 Quick start

```python
from symfm import SymFM

# Initialize with full state dimension and target active subspace dimension
model = SymFM(state_dim=40, active_subspace_dim=8)

# Pretrain the physics-informed foundation model encoder
model.pretrain(corpus="synthetic_dynamics_50k")

# Fine-tune end-to-end on partial, noisy observations
model.fit(observations=y, obs_matrix=H)

# Extract the closed-form governing equation
equation = model.extract_symbolic()
print(equation)
```

---

## 📁 Repository structure

```
SymFM/
├── symfm/
│   ├── modules/
│   │   ├── pinn_obs.py          # Module 1 — state estimation
│   │   ├── fm_encoder.py        # Module 2 — foundation model encoder
│   │   ├── active_subspace.py   # Module 3 — dimensionality reduction
│   │   └── kan_head.py          # Module 4 — symbolic regression
│   └── symfm.py                 # end-to-end pipeline
├── benchmarks/
│   ├── lorenz96/
│   ├── navier_stokes/
│   └── seird/
├── results/
├── SymFM_Paper_Final.pdf
└── README.md
```

---

## 📖 Citation

If you use SymFM in your research, please cite:

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

## 🪪 License

- **Code** — released under the [MIT License](LICENSE)
- **Paper text** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless noted otherwise

---

<div align="center">

<sub>Built with 🔥 PyTorch · 🧩 kan · ✏️ PySR · 📐 pysindy · 🌀 torchdiffeq · 📊 Weights & Biases</sub>

<br><br>

**⭐ If this project is useful to you, consider starring the repo!**

</div>
