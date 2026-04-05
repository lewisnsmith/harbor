"""Q1 Emergent Coordination — Batch Experiment Runner.

Runs 200 seeds × 2 conditions (treatment: autonomous agents, null: random agents)
and produces a summary CSV + PWS time series for analysis.

Usage:
    python experiments/q1_batch_runner.py --n-seeds 200 --output-dir results/q1_experiment --workers 8
    python experiments/q1_batch_runner.py --n-seeds 3 --workers 1 --output-dir results/q1_test  # dev
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc, linregress

# Ensure project root is importable when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hangar.agents.metrics import compute_crowding_index
from hangar.homelab.config import ExperimentConfig
from hangar.homelab.runner import ExperimentRunner

# ---------------------------------------------------------------------------
# Personality sampling
# ---------------------------------------------------------------------------

N_AGENTS = 25
N_ASSETS = 10
N_STEPS = 500
LHS_SEED = 0


def sample_personalities() -> List[Dict[str, float]]:
    """Generate 25 personality points via Latin Hypercube Sampling.

    Returns list of dicts with 'risk_appetite' and 'reactivity' keys,
    each in [0.1, 1.0].
    """
    sampler = qmc.LatinHypercube(d=2, seed=LHS_SEED)
    raw = sampler.random(n=N_AGENTS)
    personalities = []
    for x, y in raw:
        personalities.append({
            "risk_appetite": round(0.1 + 0.9 * x, 4),
            "reactivity": round(0.1 + 0.9 * y, 4),
        })
    return personalities


PERSONALITIES = sample_personalities()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_treatment_config(seed: int) -> ExperimentConfig:
    """Build config for treatment condition: 25 autonomous agents."""
    agents = []
    for i, p in enumerate(PERSONALITIES):
        agents.append({
            "type": "autonomous",
            "name": f"auto_{i}",
            "params": {"risk_appetite": p["risk_appetite"], "reactivity": p["reactivity"]},
        })
    return ExperimentConfig(
        name=f"q1_treatment_seed{seed}",
        seed=seed,
        venue={"type": "equity", "params": {"n_steps": N_STEPS, "n_assets": N_ASSETS}},
        agents=agents,
        recording={"type": "noop", "params": {}},
        evaluation={"metrics": []},
    )


def build_null_config(seed: int) -> ExperimentConfig:
    """Build config for null condition: 25 random agents."""
    agents = []
    for i in range(N_AGENTS):
        agents.append({
            "type": "random",
            "name": f"rand_{i}",
            "params": {"seed": seed * 100 + i},
        })
    return ExperimentConfig(
        name=f"q1_null_seed{seed}",
        seed=seed,
        venue={"type": "equity", "params": {"n_steps": N_STEPS, "n_assets": N_ASSETS}},
        agents=agents,
        recording={"type": "noop", "params": {}},
        evaluation={"metrics": []},
    )


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------


def extract_summary(
    result: Any,
    seed: int,
    condition: str,
) -> Tuple[Dict[str, Any], pd.Series]:
    """Extract summary stats and PWS time series from an ExperimentResult."""
    pws = compute_crowding_index(result.agent_weights)

    # PWS over final half (steps 250-500)
    mid = N_STEPS // 2
    pws_final_half = float(pws.iloc[mid:].mean()) if len(pws) > mid else float(pws.mean())

    # PWS slope via linear regression over steps 100-500
    start_slope = min(100, len(pws) - 1)
    pws_vals = pws.iloc[start_slope:].values
    if len(pws_vals) > 2:
        slope_result = linregress(np.arange(len(pws_vals)), pws_vals)
        pws_slope = float(slope_result.slope)
    else:
        pws_slope = 0.0

    summary = {
        "seed": seed,
        "condition": condition,
        "pws_final_half": pws_final_half,
        "pws_slope": pws_slope,
        "pws_mean": float(pws.mean()),
        "pws_std": float(pws.std()),
    }

    # Within-quadrant similarity (treatment only)
    if condition == "treatment":
        summary.update(_quadrant_similarity(result.agent_weights))

    return summary, pws


def _quadrant_similarity(
    agent_weights: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """Compute within-quadrant PWS for treatment agents.

    Partition 25 agents into 4 quadrants by median personality dimensions.
    """
    med_risk = np.median([p["risk_appetite"] for p in PERSONALITIES])
    med_react = np.median([p["reactivity"] for p in PERSONALITIES])

    quadrants: Dict[str, List[str]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
    for i, p in enumerate(PERSONALITIES):
        name = f"auto_{i}"
        if p["risk_appetite"] >= med_risk and p["reactivity"] >= med_react:
            quadrants["Q1"].append(name)
        elif p["risk_appetite"] < med_risk and p["reactivity"] >= med_react:
            quadrants["Q2"].append(name)
        elif p["risk_appetite"] < med_risk and p["reactivity"] < med_react:
            quadrants["Q3"].append(name)
        else:
            quadrants["Q4"].append(name)

    result = {}
    for qname, members in quadrants.items():
        if len(members) >= 2:
            subset = {k: v for k, v in agent_weights.items() if k in members}
            pws = compute_crowding_index(subset)
            result[f"pws_{qname.lower()}"] = float(pws.iloc[N_STEPS // 2:].mean())
        else:
            result[f"pws_{qname.lower()}"] = np.nan
    return result


# ---------------------------------------------------------------------------
# Single run (picklable for multiprocessing)
# ---------------------------------------------------------------------------


def run_single(args: Tuple[int, str]) -> Tuple[Dict[str, Any], pd.Series]:
    """Run one seed × condition combination. Returns (summary_dict, pws_series)."""
    seed, condition = args
    if condition == "treatment":
        config = build_treatment_config(seed)
    else:
        config = build_null_config(seed)

    result = ExperimentRunner(config).run()
    return extract_summary(result, seed, condition)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Q1 Emergent Coordination batch runner")
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/q1_experiment")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = output_dir / "pws_timeseries"
    ts_dir.mkdir(exist_ok=True)

    # Build task list: all seed × condition combinations
    tasks = []
    for seed in range(args.n_seeds):
        tasks.append((seed, "treatment"))
        tasks.append((seed, "control"))

    print(f"Running {len(tasks)} experiments ({args.n_seeds} seeds × 2 conditions)")
    print(f"Workers: {args.workers}, Output: {output_dir}")

    summaries = []
    t0 = time.time()

    if args.workers <= 1:
        # Sequential for debugging
        for i, task in enumerate(tasks):
            summary, pws = run_single(task)
            summaries.append(summary)
            seed, condition = task
            pws.to_frame("pws").to_csv(ts_dir / f"{condition}_seed{seed}.csv")
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(tasks)}] {elapsed:.1f}s elapsed")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single, task): task for task in tasks}
            done = 0
            for future in as_completed(futures):
                seed, condition = futures[future]
                summary, pws = future.result()
                summaries.append(summary)
                pws.to_frame("pws").to_csv(ts_dir / f"{condition}_seed{seed}.csv")
                done += 1
                if done % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{len(tasks)}] {elapsed:.1f}s elapsed")

    # Save summary CSV
    df = pd.DataFrame(summaries)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Summary: {csv_path} ({len(df)} rows)")
    print(f"Time series: {ts_dir}/")

    # Quick stats preview
    for cond in ["treatment", "control"]:
        subset = df[df["condition"] == cond]
        print(f"\n  {cond}: PWS final_half = {subset['pws_final_half'].mean():.4f} "
              f"± {subset['pws_final_half'].std():.4f}")


if __name__ == "__main__":
    main()
