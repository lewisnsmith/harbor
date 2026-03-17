#!/usr/bin/env python3
"""H3 Demo: What happens when 30 momentum agents trade the same market?

Produces a 4-panel figure comparing agent-influenced vs baseline markets:
  Panel 1: Synthetic price paths (with/without agents)
  Panel 2: Crowding index over time
  Panel 3: Rolling return autocorrelation comparison
  Panel 4: Rolling volatility comparison

Output: results/agent_simulation/demo_figure.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harbor.agents import (
    AgentConfig,
    MarketConfig,
    MarketEnvironment,
    MomentumAgent,
    compute_crowding_index,
    compute_return_autocorrelation,
    run_simulation,
)


def main(output_dir: str = "results/agent_simulation", n_steps: int = 500) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running H3 agent simulation demo ({n_steps} steps)...")

    # --- Shared market config ---
    base_config = MarketConfig(
        n_steps=n_steps,
        n_assets=10,
        base_volatility=0.01,
        base_drift=0.0002,
        temporary_impact=0.1,
        permanent_impact=0.01,
        correlation=0.3,
        seed=42,
    )

    # --- Baseline: no agents ---
    print("  Running baseline (no agents)...")
    baseline_market = MarketEnvironment(base_config)
    baseline_result = run_simulation(baseline_market, agents=[])

    # --- Agent market: 30 momentum agents ---
    print("  Running agent market (30 momentum agents)...")
    agent_market = MarketEnvironment(base_config)
    agents = [
        MomentumAgent(AgentConfig(f"mom_{i}"), n_assets=10, lookback=21)
        for i in range(30)
    ]
    agent_result = run_simulation(agent_market, agents)

    # --- Compute metrics ---
    print("  Computing metrics...")
    crowding = compute_crowding_index(agent_result.agent_weights)

    baseline_autocorr = compute_return_autocorrelation(
        baseline_result.returns, lag=1, window=63
    )
    agent_autocorr = compute_return_autocorrelation(
        agent_result.returns, lag=1, window=63
    )

    # Rolling volatility (21-day)
    baseline_vol = baseline_result.returns.mean(axis=1).rolling(21).std() * np.sqrt(252)
    agent_vol = agent_result.returns.mean(axis=1).rolling(21).std() * np.sqrt(252)

    # --- Plot ---
    print("  Generating figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "H3: Agent-Influenced vs Baseline Market Dynamics\n"
        "(30 Momentum Agents with Price Impact)",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Price paths
    ax1 = axes[0, 0]
    # Plot equal-weighted index
    baseline_idx = (1 + baseline_result.returns.mean(axis=1)).cumprod()
    agent_idx = (1 + agent_result.returns.mean(axis=1)).cumprod()
    ax1.plot(baseline_idx.values, label="Baseline (no agents)", alpha=0.8)
    ax1.plot(agent_idx.values, label="30 Momentum Agents", alpha=0.8)
    ax1.set_title("Equal-Weighted Price Index")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Crowding index
    ax2 = axes[0, 1]
    ax2.plot(crowding.values, color="darkred", alpha=0.8)
    ax2.axhline(crowding.mean(), ls="--", color="gray", alpha=0.6, label=f"Mean: {crowding.mean():.3f}")
    ax2.set_title("Crowding Index (Pairwise Weight Similarity)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cosine Similarity")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Panel 3: Return autocorrelation
    ax3 = axes[1, 0]
    ax3.plot(baseline_autocorr.values, label="Baseline", alpha=0.7)
    ax3.plot(agent_autocorr.values, label="30 Mom Agents", alpha=0.7)
    ax3.axhline(0, ls="-", color="black", alpha=0.3)
    ax3.set_title("Rolling Return Autocorrelation (63-day)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Autocorrelation (lag-1)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling volatility
    ax4 = axes[1, 1]
    ax4.plot(baseline_vol.values, label="Baseline", alpha=0.7)
    ax4.plot(agent_vol.values, label="30 Mom Agents", alpha=0.7)
    ax4.set_title("Rolling Annualized Volatility (21-day)")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Volatility")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_path / "demo_figure.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Figure saved: {fig_path}")

    # --- Print summary ---
    from harbor.agents import compute_simulation_summary

    baseline_summary = compute_simulation_summary(
        baseline_result.prices, baseline_result.returns, {}, baseline_result.orders
    )
    agent_summary = compute_simulation_summary(
        agent_result.prices,
        agent_result.returns,
        agent_result.agent_weights,
        agent_result.orders,
    )

    print("\n  === Summary Metrics ===")
    print(f"  {'Metric':<30s} {'Baseline':>12s} {'30 Mom Agents':>14s}")
    print(f"  {'-'*58}")
    for key in ["annualized_vol", "return_autocorrelation", "crowding_mean", "flow_imbalance_mean", "regime_shock_count"]:
        bv = baseline_summary.get(key, "N/A")
        av = agent_summary.get(key, "N/A")
        bv_str = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        av_str = f"{av:.4f}" if isinstance(av, float) else str(av)
        print(f"  {key:<30s} {bv_str:>12s} {av_str:>14s}")

    print("\n  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3 Agent Simulation Demo")
    parser.add_argument(
        "--output-dir",
        default="results/agent_simulation",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--n-steps", type=int, default=500, help="Number of simulation steps"
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, n_steps=args.n_steps)
