"""Multi-agent simulation orchestrator.

Pipeline:
1) Load prices, scan checkpoints to build AgentSpec list
2) run_multi_agent_simulation()
3) Extract per-agent performance_summary, crowding_proxy, correlation_series
4) Generate figures: cumulative returns, crowding time series, weight similarity
5) Save all outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hangar.backtest.metrics import performance_summary  # noqa: E402
from hangar.data import load_sp500_prices, load_sp500_tickers  # noqa: E402
from hangar.ml.behavior_agents import (  # noqa: E402
    AgentSpec,
    run_multi_agent_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent portfolio simulation.")
    parser.add_argument(
        "--checkpoint-dir",
        default="data/models/rl_agents",
        help="Directory with per-agent checkpoint subdirectories",
    )
    parser.add_argument("--start", default="2020-01-01", help="Simulation start date")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="Simulation end date",
    )
    parser.add_argument("--max-assets", type=int, default=20, help="Max tickers")
    parser.add_argument("--output-dir", default="output/rl_simulation", help="Output directory")
    return parser.parse_args()


def find_latest_checkpoint(agent_dir: Path) -> str | None:
    """Find the latest RLlib checkpoint in an agent directory."""
    candidates = sorted(agent_dir.glob("checkpoint_*"), reverse=True)
    if candidates:
        return str(candidates[0])
    # Also check for nested checkpoint dirs
    for sub in sorted(agent_dir.iterdir(), reverse=True):
        if sub.is_dir():
            nested = sorted(sub.glob("checkpoint_*"), reverse=True)
            if nested:
                return str(nested[0])
    return None


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print("[1/5] Loading prices...")
    tickers = load_sp500_tickers(as_of=args.end)[: args.max_assets]
    prices = load_sp500_prices(tickers=tickers, start=args.start, end=args.end, adjusted=True)
    returns = prices.pct_change().dropna(how="all")
    min_obs = max(int(0.8 * len(returns)), 60)
    returns = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)
    print(f"  {returns.shape[1]} assets, {returns.shape[0]} days")

    # 2) Scan checkpoints
    print("[2/5] Scanning checkpoints...")
    ckpt_base = Path(args.checkpoint_dir)
    agent_specs = []
    for agent_dir in sorted(ckpt_base.iterdir()):
        if not agent_dir.is_dir():
            continue
        ckpt_path = find_latest_checkpoint(agent_dir)
        if ckpt_path is None:
            print(f"  Skipping {agent_dir.name}: no checkpoint found")
            continue
        agent_specs.append(
            AgentSpec(
                name=agent_dir.name,
                checkpoint_path=ckpt_path,
            )
        )
        print(f"  Found: {agent_dir.name} -> {ckpt_path}")

    if len(agent_specs) < 2:
        print("Need at least 2 agents for multi-agent simulation.")
        return

    # 3) Run simulation
    print(f"[3/5] Running multi-agent simulation with {len(agent_specs)} agents...")
    result = run_multi_agent_simulation(
        returns,
        agent_specs,
        start_idx=60,  # skip warmup
    )

    # 4) Extract performance metrics
    print("[4/5] Computing metrics and generating figures...")
    perf = {}
    for name, ret_series in result.agent_returns.items():
        clean = ret_series.dropna()
        if len(clean) > 10:
            perf[name] = performance_summary(clean)
        else:
            perf[name] = {}

    # Print summary
    for name, metrics in perf.items():
        sharpe = metrics.get("sharpe", float("nan"))
        mdd = metrics.get("max_drawdown", float("nan"))
        print(f"  {name:>20s}: Sharpe={sharpe:.3f}  MaxDD={mdd:.3f}")

    # Figure 1: Cumulative returns
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    for name, ret_series in result.agent_returns.items():
        cum = (1 + ret_series.fillna(0)).cumprod()
        ax1.plot(cum.index, cum.values, label=name, linewidth=0.8)
    ax1.set_title("Multi-Agent Cumulative Returns")
    ax1.set_ylabel("Wealth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(fig_dir / "cumulative_returns.png", dpi=150, bbox_inches="tight")

    # Figure 2: Crowding proxy
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(
        result.crowding_proxy.index,
        result.crowding_proxy.values,
        color="purple",
        linewidth=0.8,
    )
    ax2.set_title("Weight Similarity (Crowding Proxy)")
    ax2.set_ylabel("Avg Cosine Similarity")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / "crowding_proxy.png", dpi=150, bbox_inches="tight")

    # Figure 3: Correlation time series
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(
        result.correlation_series.index,
        result.correlation_series.values,
        color="teal",
        linewidth=0.8,
    )
    ax3.set_title("Avg Pairwise Return Correlation (21-day rolling)")
    ax3.set_ylabel("Correlation")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(fig_dir / "return_correlation.png", dpi=150, bbox_inches="tight")

    plt.close("all")

    # 5) Save outputs
    print("[5/5] Saving outputs...")

    with open(out / "performance_summary.json", "w") as f:
        json.dump(perf, f, indent=2, default=str)

    result.crowding_proxy.to_csv(out / "crowding_proxy.csv", header=True)
    result.correlation_series.to_csv(out / "correlation_series.csv", header=True)

    for name, w_df in result.agent_weights.items():
        w_df.to_csv(out / f"weights_{name}.csv")

    print(f"  Outputs saved to {out}")
    print("Done.")


if __name__ == "__main__":
    main()
