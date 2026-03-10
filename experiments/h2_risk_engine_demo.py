#!/usr/bin/env python3
"""Phase H2 exit-criteria pipeline: advanced risk engine demo.

Demonstrates:
1. Regime-aware covariance estimation
2. Non-Gaussian (Student-t) Monte Carlo simulation
3. Config-driven stress scenario suite
4. Risk decomposition and concentration metrics

Usage:
    python experiments/h2_risk_engine_demo.py --start 2015-01-01 --max-assets 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Ensure harbor is importable when running from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harbor.data import load_sp500_tickers, load_prices
from harbor.risk.covariance import estimate_covariance, expanding_regime_covariance
from harbor.risk.decomposition import (
    cluster_risk_attribution,
    component_risk,
    concentration_metrics,
    percent_risk_contribution,
)
from harbor.risk.engine import RiskConfig, RiskEngine, load_scenarios_config
from harbor.risk.monte_carlo import (
    monte_carlo_var_cvar_from_history,
    simulate_student_t_returns,
)
from harbor.risk.scenarios import run_scenario_suite, scenario_report_to_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H2 Risk Engine Demo")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None, help="End date (default: today)")
    p.add_argument("--max-assets", type=int, default=20)
    p.add_argument("--output-dir", default="results/h2_risk")
    p.add_argument(
        "--simulation-method",
        choices=["normal", "student_t"],
        default="normal",
    )
    p.add_argument(
        "--scenarios-config",
        default=os.path.join(
            os.path.dirname(__file__), "..", "configs", "risk", "scenarios.json"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading S&P 500 tickers and prices ...")
    tickers = load_sp500_tickers()
    if len(tickers) > args.max_assets:
        tickers = tickers[: args.max_assets]

    prices = load_prices(tickers, start=args.start, end=end)
    returns = prices.pct_change().dropna(how="all").iloc[1:]
    returns = returns.dropna(axis=1, how="all")
    print(f"  Universe: {returns.shape[1]} assets, {returns.shape[0]} days")

    # Equal weights for demo
    assets = list(returns.columns)
    weights = pd.Series(1.0 / len(assets), index=assets)
    mean_returns = returns.mean()

    # ------------------------------------------------------------------
    # 2. Regime-aware covariance
    # ------------------------------------------------------------------
    print("\nEstimating regime-aware covariance ...")
    regime_covs = expanding_regime_covariance(returns, vol_threshold_pct=0.8)
    for regime, cov in regime_covs.items():
        print(f"  {regime}: {cov.shape[0]}x{cov.shape[1]} matrix")

    # Use conservative (high-vol) for stress, standard for baseline
    baseline_cov = estimate_covariance(returns, method="ledoit_wolf", annualization=1)
    high_vol_cov = regime_covs.get("high_vol", baseline_cov)

    # ------------------------------------------------------------------
    # 3. Baseline vs Student-t VaR/CVaR comparison
    # ------------------------------------------------------------------
    print("\nComparing Normal vs Student-t Monte Carlo ...")
    normal_result = monte_carlo_var_cvar_from_history(
        returns, weights, n_sims=10_000, horizon=21, random_state=42,
        simulation_method="normal",
    )
    student_result = monte_carlo_var_cvar_from_history(
        returns, weights, n_sims=10_000, horizon=21, random_state=42,
        simulation_method="student_t", simulation_kwargs={"df": 5},
    )
    print(f"  Normal:    VaR={normal_result.var:.4f}  CVaR={normal_result.cvar:.4f}")
    print(f"  Student-t: VaR={student_result.var:.4f}  CVaR={student_result.cvar:.4f}")

    mc_comparison = {
        "normal": {"var": normal_result.var, "cvar": normal_result.cvar, "expected_return": normal_result.expected_return},
        "student_t_df5": {"var": student_result.var, "cvar": student_result.cvar, "expected_return": student_result.expected_return},
    }
    with open(os.path.join(args.output_dir, "mc_comparison.json"), "w") as f:
        json.dump(mc_comparison, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Stress scenario suite
    # ------------------------------------------------------------------
    print("\nRunning stress scenarios ...")

    # Simple sector map (assign sectors round-robin for demo)
    sector_names = ["Technology", "Energy", "Healthcare", "Financials"]
    sector_map = {asset: sector_names[i % len(sector_names)] for i, asset in enumerate(assets)}

    if os.path.exists(args.scenarios_config):
        scenarios = load_scenarios_config(args.scenarios_config)
    else:
        scenarios = [
            {"name": "vol_spike_2x", "type": "vol_spike", "params": {"multiplier": 2.0}},
            {"name": "corr_spike_90", "type": "correlation_spike", "params": {"target_corr": 0.90}},
        ]

    results = run_scenario_suite(
        weights, mean_returns, baseline_cov,
        scenarios,
        sector_map=sector_map,
        n_sims=10_000, horizon=21, alpha=0.95, random_state=42,
    )

    report = scenario_report_to_dict(results)
    with open(os.path.join(args.output_dir, "stress_report.json"), "w") as f:
        # Remove stressed_cov from JSON output (too large)
        slim_report = []
        for entry in report:
            slim = {k: v for k, v in entry.items() if k != "stressed_cov"}
            slim_report.append(slim)
        json.dump(slim_report, f, indent=2)

    for r in results:
        print(f"  {r.name}: VaR {r.baseline_var:.4f} -> {r.stressed_var:.4f} ({r.var_change_pct:+.1f}%)")

    # ------------------------------------------------------------------
    # 5. Risk decomposition
    # ------------------------------------------------------------------
    print("\nRisk decomposition ...")
    prc = percent_risk_contribution(weights, baseline_cov)
    cr = component_risk(weights, baseline_cov)
    metrics = concentration_metrics(weights, baseline_cov)

    decomp_df = pd.DataFrame({
        "weight": weights,
        "component_risk": cr,
        "risk_pct": prc,
    })
    decomp_df.to_csv(os.path.join(args.output_dir, "risk_decomposition.csv"))

    cluster_df = cluster_risk_attribution(weights, baseline_cov, sector_map)
    cluster_df.to_csv(os.path.join(args.output_dir, "cluster_attribution.csv"), index=False)

    metrics_out = {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in metrics.items()}
    with open(os.path.join(args.output_dir, "concentration_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"  Effective N (weight): {metrics['effective_n_weight']:.1f}")
    print(f"  Effective N (risk):   {metrics['effective_n_risk']:.1f}")
    print(f"  Top risk contributor: {metrics['max_risk_contributor']} ({metrics['max_risk_pct']:.1%})")

    # ------------------------------------------------------------------
    # 6. Risk engine end-to-end
    # ------------------------------------------------------------------
    print("\nRisk engine end-to-end (pluggable interface) ...")
    sim_method = args.simulation_method
    sim_kwargs = {"df": 5} if sim_method == "student_t" else {}
    config = RiskConfig(
        covariance_method="ledoit_wolf",
        simulation_method=sim_method,
        simulation_kwargs=sim_kwargs,
        n_sims=10_000, horizon=21, alpha=0.95,
    )
    engine = RiskEngine(config)
    var_cvar = engine.compute_var_cvar(weights, mean_returns, baseline_cov, random_state=42)
    print(f"  VaR: {var_cvar.var:.4f}  CVaR: {var_cvar.cvar:.4f}")

    decomp = engine.decompose_risk(weights, baseline_cov)
    print(f"  Concentration (effective N): {decomp['concentration_metrics']['effective_n_bets']:.1f}")

    print(f"\nOutputs saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
