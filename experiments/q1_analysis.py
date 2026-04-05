"""Q1 Emergent Coordination — Statistical Analysis and Figure Generation.

Reads the summary CSV from q1_batch_runner.py, runs statistical tests,
and produces publication-quality figures.

Usage:
    python experiments/q1_analysis.py --input results/q1_experiment --output results/q1_experiment/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled standard deviation)."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI of the difference in means (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        diffs[i] = a_boot.mean() - b_boot.mean()
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {"ci_lower": lo, "ci_upper": hi, "mean_diff": float(diffs.mean())}


def run_statistics(summary: pd.DataFrame) -> dict:
    """Run all statistical tests on the summary data."""
    treat = summary[summary["condition"] == "treatment"]["pws_final_half"].values
    ctrl = summary[summary["condition"] == "control"]["pws_final_half"].values

    # Primary: one-sided two-sample t-test
    t_stat, p_two = stats.ttest_ind(treat, ctrl)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    result = {
        "primary_test": {
            "metric": "pws_final_half",
            "treatment_mean": float(treat.mean()),
            "treatment_std": float(treat.std()),
            "control_mean": float(ctrl.mean()),
            "control_std": float(ctrl.std()),
            "t_statistic": float(t_stat),
            "p_value_one_sided": float(p_one),
            "cohens_d": cohens_d(treat, ctrl),
            "n_treatment": len(treat),
            "n_control": len(ctrl),
        },
    }

    # Bootstrap CI
    boot = bootstrap_ci(treat, ctrl)
    result["primary_test"].update(boot)

    # Secondary: convergence slope
    treat_slope = summary[summary["condition"] == "treatment"]["pws_slope"].values
    ctrl_slope = summary[summary["condition"] == "control"]["pws_slope"].values
    t_slope, p_slope_two = stats.ttest_ind(treat_slope, ctrl_slope)
    p_slope_one = p_slope_two / 2 if t_slope > 0 else 1 - p_slope_two / 2

    result["slope_test"] = {
        "metric": "pws_slope",
        "treatment_mean": float(treat_slope.mean()),
        "control_mean": float(ctrl_slope.mean()),
        "t_statistic": float(t_slope),
        "p_value_one_sided": float(p_slope_one),
        "cohens_d": cohens_d(treat_slope, ctrl_slope),
    }

    # Secondary: within-quadrant (treatment only)
    treat_only = summary[summary["condition"] == "treatment"]
    quadrant_stats = {}
    for q in ["pws_q1", "pws_q2", "pws_q3", "pws_q4"]:
        if q in treat_only.columns:
            vals = treat_only[q].dropna().values
            if len(vals) > 0:
                quadrant_stats[q] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                }
    result["quadrant_similarity"] = quadrant_stats

    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def load_pws_timeseries(ts_dir: Path, n_seeds: int) -> dict:
    """Load PWS time series from CSV files."""
    series = {"treatment": [], "control": []}
    for cond in ["treatment", "control"]:
        for seed in range(n_seeds):
            path = ts_dir / f"{cond}_seed{seed}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col=0)
                series[cond].append(df["pws"].values)
    return series


def plot_pws_timeseries(ts_data: dict, output_path: Path, n_steps: int = 500) -> None:
    """Plot mean ± 1 SE of PWS over time, treatment vs control."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for cond, color, label in [("treatment", "#2563eb", "Autonomous (treatment)"),
                                ("control", "#6b7280", "Random (control)")]:
        if not ts_data[cond]:
            continue
        arr = np.array(ts_data[cond])
        mean = arr.mean(axis=0)
        se = arr.std(axis=0) / np.sqrt(arr.shape[0])
        steps = np.arange(len(mean))
        ax.plot(steps, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.2)

    # Vertical line at midpoint
    mid = n_steps // 2
    ax.axvline(x=mid, color="#dc2626", linestyle="--", alpha=0.5, label=f"Step {mid} (analysis cutoff)")

    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Pairwise Weight Similarity (PWS)")
    ax.set_title("Q1: Emergent Coordination — PWS Over Time")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_distribution(summary: pd.DataFrame, output_path: Path) -> None:
    """Histogram/KDE overlay of pws_final_half for treatment vs control."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond, color, label in [("treatment", "#2563eb", "Autonomous (treatment)"),
                                ("control", "#6b7280", "Random (control)")]:
        vals = summary[summary["condition"] == cond]["pws_final_half"].values
        ax.hist(vals, bins=25, alpha=0.4, color=color, label=label, density=True)
        # KDE
        if len(vals) > 2:
            kde = stats.gaussian_kde(vals)
            x = np.linspace(vals.min() - 0.01, vals.max() + 0.01, 200)
            ax.plot(x, kde(x), color=color, linewidth=2)

    ax.set_xlabel("Mean PWS (steps 250-500)")
    ax.set_ylabel("Density")
    ax.set_title("Q1: Distribution of Coordination Metric")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_personality_scatter(summary: pd.DataFrame, output_path: Path) -> None:
    """Scatter of 25 LHS personality points colored by mean PWS contribution."""
    from experiments.q1_batch_runner import PERSONALITIES

    fig, ax = plt.subplots(figsize=(7, 6))

    # Average pws_final_half across seeds for treatment
    treat = summary[summary["condition"] == "treatment"]
    mean_pws = treat["pws_final_half"].mean()

    # Color by personality position (approximation — we don't have per-agent PWS,
    # so color by risk_appetite * reactivity as a proxy for activity level)
    risk = [p["risk_appetite"] for p in PERSONALITIES]
    react = [p["reactivity"] for p in PERSONALITIES]
    activity = [r * re for r, re in zip(risk, react)]

    scatter = ax.scatter(risk, react, c=activity, cmap="RdYlBu_r", s=120,
                         edgecolors="black", linewidth=0.5, zorder=5)
    plt.colorbar(scatter, ax=ax, label="Risk × Reactivity")

    for i, p in enumerate(PERSONALITIES):
        ax.annotate(str(i), (p["risk_appetite"], p["reactivity"]),
                    fontsize=7, ha="center", va="center")

    ax.set_xlabel("Risk Appetite")
    ax.set_ylabel("Reactivity")
    ax.set_title(f"Q1: Agent Personality Space (25 LHS points)\nMean PWS = {mean_pws:.4f}")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Q1 analysis and figure generation")
    parser.add_argument("--input", type=str, default="results/q1_experiment")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run q1_batch_runner.py first.")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    n_seeds = summary["seed"].nunique()
    print(f"Loaded {len(summary)} rows ({n_seeds} seeds × 2 conditions)")

    # Run statistics
    print("\n--- Statistical Tests ---")
    result = run_statistics(summary)

    primary = result["primary_test"]
    print(f"  Treatment PWS: {primary['treatment_mean']:.4f} ± {primary['treatment_std']:.4f}")
    print(f"  Control PWS:   {primary['control_mean']:.4f} ± {primary['control_std']:.4f}")
    print(f"  t = {primary['t_statistic']:.3f}, p = {primary['p_value_one_sided']:.2e} (one-sided)")
    print(f"  Cohen's d = {primary['cohens_d']:.3f}")
    print(f"  95% CI of difference: [{primary['ci_lower']:.4f}, {primary['ci_upper']:.4f}]")

    slope = result["slope_test"]
    print(f"\n  Slope test: t = {slope['t_statistic']:.3f}, p = {slope['p_value_one_sided']:.2e}")

    if result["quadrant_similarity"]:
        print("\n  Within-quadrant PWS (treatment):")
        for q, v in result["quadrant_similarity"].items():
            print(f"    {q}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Save stats
    stats_path = input_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # Generate figures
    print("\n--- Figures ---")

    # 1. PWS time series
    ts_dir = input_dir / "pws_timeseries"
    if ts_dir.exists():
        ts_data = load_pws_timeseries(ts_dir, n_seeds)
        plot_pws_timeseries(ts_data, output_dir / "pws_timeseries.png")
    else:
        print("  WARN: No time series directory found, skipping time series plot")

    # 2. Distribution
    plot_distribution(summary, output_dir / "distribution.png")

    # 3. Personality scatter
    plot_personality_scatter(summary, output_dir / "personality_scatter.png")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
