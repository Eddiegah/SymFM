"""
scalability_figure.py
---------------------
Generates the scalability figure for the SymFM paper.

Two panels:
  Panel (a): L2 Error vs state dimension N for SymFM and SINDy
  Panel (b): Runtime vs state dimension N for SymFM and SINDy

Uses existing results from the results/ folder.

Author: Edmund Eric Gah
Project: SymFM
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load results ──────────────────────────────────────────────────────────────

def load_results(results_dir="results"):
    dims   = [4, 10, 20, 40]
    data   = {}
    for method in ["symfm", "sindy", "kan"]:
        data[method] = {"N": [], "L2": [], "ERR": [], "runtime": []}
        for N in dims:
            path = os.path.join(results_dir, f"{method}_N{N}.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                data[method]["N"].append(N)
                data[method]["L2"].append(r.get("L2_mean", 10.0))
                data[method]["ERR"].append(r.get("ERR_mean", 0.0))
                data[method]["runtime"].append(r.get("runtime_mean_s", 0.0))
    return data

# ── Generate figure ───────────────────────────────────────────────────────────

def generate_scalability_figure(results_dir="results"):
    data = load_results(results_dir)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    fig.suptitle(
        "SymFM Scalability Analysis: Lorenz-96 Benchmark",
        fontsize=14, fontweight='bold', y=1.02
    )

    colors = {
        'symfm': '#4CAF50',
        'sindy': '#2196F3',
        'kan':   '#FF9800',
    }
    labels = {
        'symfm': 'SymFM (ours)',
        'sindy': 'SINDy',
        'kan':   'KAN flat',
    }
    markers = {
        'symfm': 'o',
        'sindy': 's',
        'kan':   '^',
    }

    # ── Panel (a): L2 Error vs N ──────────────────────────────────────────────
    for method in ['symfm', 'sindy', 'kan']:
        d = data[method]
        if not d["N"]:
            continue
        ax1.plot(d["N"], d["L2"],
                 color=colors[method],
                 marker=markers[method],
                 linewidth=2.5,
                 markersize=9,
                 label=labels[method],
                 zorder=3)
        for x, y in zip(d["N"], d["L2"]):
            ax1.annotate(f'{y:.3f}',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8,
                         color=colors[method])

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.4,
                linewidth=1.5, label='L2=1.0 (uninformative)')
    ax1.set_xlabel("State Dimension $N$", fontsize=12)
    ax1.set_ylabel("Relative $\\ell_2$ Error", fontsize=12)
    ax1.set_title("(a) Derivative Approximation Error vs $N$", fontsize=12)
    ax1.set_xticks([4, 10, 20, 40])
    ax1.set_xticklabels(["N=4", "N=10", "N=20", "N=40"])
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 2.2)

    # Add shaded region showing where SINDy fails
    ax1.axvspan(15, 45, alpha=0.05, color='red',
                label='High-dim regime (SINDy fails)')
    ax1.text(27, 2.0, "SINDy fails\n(ERR=0%)",
             ha='center', fontsize=9, color='red', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(7, 0.05, "Both\nrecover\n(ERR=100%)",
             ha='center', fontsize=8, color='green', alpha=0.8)

    # ── Panel (b): ERR vs N ───────────────────────────────────────────────────
    for method in ['symfm', 'sindy', 'kan']:
        d = data[method]
        if not d["N"]:
            continue
        ax2.plot(d["N"], d["ERR"],
                 color=colors[method],
                 marker=markers[method],
                 linewidth=2.5,
                 markersize=9,
                 label=labels[method],
                 zorder=3)
        for x, y in zip(d["N"], d["ERR"]):
            ax2.annotate(f'{y:.0f}%',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8,
                         color=colors[method])

    ax2.set_xlabel("State Dimension $N$", fontsize=12)
    ax2.set_ylabel("Equation Recovery Rate (%)", fontsize=12)
    ax2.set_title("(b) Equation Recovery Rate vs $N$", fontsize=12)
    ax2.set_xticks([4, 10, 20, 40])
    ax2.set_xticklabels(["N=4", "N=10", "N=20", "N=40"])
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 115)
    ax2.axhline(y=100, color='green', linestyle='--',
                alpha=0.3, linewidth=1.5)
    ax2.axvspan(15, 45, alpha=0.05, color='red')

    # Save
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "scalability_figure.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_path}")
    return save_path


if __name__ == "__main__":
    print("=" * 60)
    print("SymFM: Generating Scalability Figure")
    print("=" * 60)
    path = generate_scalability_figure("results")
    print(f"Done. Open: {path}")