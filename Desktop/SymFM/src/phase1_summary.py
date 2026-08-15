"""
phase1_summary.py
-----------------
Generates the Phase 1 summary comparing SINDy and KAN baselines
on Lorenz-96 at N=4 and N=10.

Produces:
  1. A clean comparison table printed to terminal
  2. A matplotlib figure saved to results/phase1_comparison.png
  3. A combined JSON summary saved to results/phase1_summary.json

This is the first set of real numbers for the SymFM paper.
These go into Table 4 (Main Results) as the baseline rows.

Author: Edmund Eric Gah
Project: SymFM
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Load results ──────────────────────────────────────────────────────────────

def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_all_results(results_dir="results"):
    results = {}
    for method in ["sindy", "kan"]:
        for N in [4, 10]:
            path = os.path.join(results_dir, f"{method}_N{N}.json")
            key  = f"{method}_N{N}"
            data = load_result(path)
            if data:
                results[key] = data
            else:
                print(f"  Warning: {path} not found, skipping.")
    return results


# ── Print comparison table ────────────────────────────────────────────────────

def print_comparison_table(results):
    print("\n" + "=" * 75)
    print("PHASE 1 BASELINE COMPARISON -- Lorenz-96")
    print("=" * 75)
    print(f"{'Method':<15} {'N':<6} {'ERR (%)':<12} {'L2 Error':<12} "
          f"{'Complexity':<12} {'Time (s)'}")
    print("─" * 75)

    methods_order = [
        ("sindy", "SINDy",    [4, 10]),
        ("kan",   "KAN flat", [4, 10]),
    ]

    for method_key, method_name, dims in methods_order:
        for N in dims:
            key = f"{method_key}_N{N}"
            if key not in results:
                print(f"{method_name:<15} {N:<6} {'N/A':<12} {'N/A':<12} "
                      f"{'N/A':<12} {'N/A'}")
                continue
            r = results[key]
            print(f"{method_name:<15} "
                  f"{r['N']:<6} "
                  f"{r['ERR_mean']:<12.1f} "
                  f"{r['L2_mean']:<12.4f} "
                  f"{r.get('complexity_mean', 0):<12.1f} "
                  f"{r['runtime_mean_s']:.3f}")
        print()

    print("=" * 75)
    print("ERR = Equation Recovery Rate (%, higher is better)")
    print("L2  = Relative L2 derivative prediction error (lower is better)")
    print("=" * 75)


# ── Generate comparison figure ────────────────────────────────────────────────

def generate_figure(results, save_dir="results"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Phase 1 Baseline Comparison: Lorenz-96\n"
        "(SymFM results will be added in Phase 3)",
        fontsize=13, fontweight='bold'
    )

    dims   = [4, 10]
    x      = np.arange(len(dims))
    width  = 0.35

    colors = {
        'sindy': '#2196F3',
        'kan':   '#FF9800',
    }
    labels = {
        'sindy': 'SINDy',
        'kan':   'KAN flat',
    }

    # Panel 1: Equation Recovery Rate
    ax1 = axes[0]
    for i, (method_key, color) in enumerate(colors.items()):
        errs = []
        for N in dims:
            key = f"{method_key}_N{N}"
            if key in results:
                errs.append(results[key]['ERR_mean'])
            else:
                errs.append(0.0)
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, errs, width,
                       label=labels[method_key],
                       color=color, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, errs):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f'{val:.0f}%',
                     ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel("State Dimension N", fontsize=11)
    ax1.set_ylabel("Equation Recovery Rate (%)", fontsize=11)
    ax1.set_title("Equation Recovery Rate", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N={d}" for d in dims])
    ax1.set_ylim(0, 120)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=100, color='green', linestyle='--',
                alpha=0.4, label='Perfect (100%)')

    # Add placeholder bar for SymFM
    ax1.text(0.98, 0.95,
             "SymFM results\nto be added\nin Phase 3",
             transform=ax1.transAxes,
             ha='right', va='top',
             fontsize=8,
             color='gray',
             style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.8))

    # Panel 2: Relative L2 Error
    ax2 = axes[1]
    for i, (method_key, color) in enumerate(colors.items()):
        l2s = []
        for N in dims:
            key = f"{method_key}_N{N}"
            if key in results:
                l2s.append(results[key]['L2_mean'])
            else:
                l2s.append(0.0)
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, l2s, width,
                       label=labels[method_key],
                       color=color, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, l2s):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{val:.3f}',
                     ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel("State Dimension N", fontsize=11)
    ax2.set_ylabel("Relative L2 Error", fontsize=11)
    ax2.set_title("Relative L2 Derivative Error", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N={d}" for d in dims])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0.98, 0.95,
             "SymFM results\nto be added\nin Phase 3",
             transform=ax2.transAxes,
             ha='right', va='top',
             fontsize=8,
             color='gray',
             style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow',
                       alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(save_dir, "phase1_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {save_path}")
    return save_path


# ── Save combined summary ─────────────────────────────────────────────────────

def save_summary(results, save_dir="results"):
    summary = {
        "phase": 1,
        "description": "Lorenz-96 baseline comparison at N=4 and N=10",
        "methods": ["SINDy", "KAN_flat"],
        "dimensions": [4, 10],
        "key_finding": (
            "SINDy recovers equations perfectly at N=4 and N=10 "
            "due to low dimensionality and sparse structure. "
            "Flat KAN struggles without active subspace projection. "
            "SymFM results to be added after Phase 3 implementation."
        ),
        "results": results,
        "paper_note": (
            "These are the baseline rows for Table 4 (Main Results) "
            "in the SymFM paper. SymFM results at N=20 and N=40 will "
            "demonstrate the advantage of the active subspace projection."
        )
    }
    save_path = os.path.join(save_dir, "phase1_summary.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {save_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("SymFM Phase 1: Summary and Comparison")
    print("=" * 60)

    results = load_all_results("results")

    if not results:
        print("No results found. Please run sindy_baseline.py and "
              "kan_baseline.py first.")
        exit(1)

    print(f"Loaded {len(results)} result files.")

    # Print table
    print_comparison_table(results)

    # Generate figure
    generate_figure(results, "results")

    # Save summary
    summary = save_summary(results, "results")

    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("\nWhat you have so far:")
    print("  data/lorenz96_N4.npz      -- Lorenz-96 data, N=4")
    print("  data/lorenz96_N10.npz     -- Lorenz-96 data, N=10")
    print("  results/sindy_N4.json     -- SINDy baseline, N=4")
    print("  results/sindy_N10.json    -- SINDy baseline, N=10")
    print("  results/kan_N4.json       -- KAN baseline, N=4")
    print("  results/kan_N10.json      -- KAN baseline, N=10")
    print("  results/phase1_comparison.png  -- Comparison figure")
    print("  results/phase1_summary.json    -- Combined summary")
    print("\nNext steps:")
    print("  Phase 2: Extend to N=20 and N=40")
    print("  Phase 3: Build SymFM with active subspace projection")
    print("  Phase 4: Full benchmark evaluation")
    print("\nThese baseline numbers go into Table 4 of your paper.")
    print("=" * 60)