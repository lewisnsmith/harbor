"""Phase H1 exit-criteria script.

Pipeline:
1) Load S&P 500 universe
2) Download historical prices
3) Build HRP weights on a rolling basis
4) Run transaction-cost-aware backtest
5) Report risk metrics + Monte Carlo VaR/CVaR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution without requiring `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hangar.backtest import run_cross_sectional_backtest  # noqa: E402
from hangar.data import load_sp500_prices, load_sp500_tickers  # noqa: E402
from hangar.portfolio import hrp_weights  # noqa: E402
from hangar.risk import estimate_covariance, monte_carlo_var_cvar_from_history  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase H1 HRP backtest pipeline.")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--max-assets", type=int, default=75, help="Max constituents to include")
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Lookback window for HRP covariance",
    )
    parser.add_argument(
        "--rebalance",
        type=int,
        default=21,
        help="Rebalance frequency (trading days)",
    )
    parser.add_argument(
        "--tcost-bps",
        type=float,
        default=5.0,
        help="One-way transaction cost in bps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = load_sp500_tickers(as_of=args.end)[: args.max_assets]
    if len(tickers) < 10:
        raise RuntimeError("Universe loader returned too few assets for a meaningful HRP backtest.")

    prices = load_sp500_prices(tickers=tickers, start=args.start, end=args.end, adjusted=True)
    returns = prices.pct_change().dropna(how="all")

    # Keep liquid/continuous names to avoid repeated missing-data distortions.
    min_obs = max(int(0.8 * len(returns)), args.lookback)
    returns = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)

    if returns.shape[0] <= args.lookback:
        raise RuntimeError(
            "Insufficient history after cleaning to run the requested lookback window."
        )

    def weight_model(window: pd.DataFrame, _current_weights: pd.Series) -> pd.Series:
        cov = estimate_covariance(window, method="ledoit_wolf", annualization=1)
        return hrp_weights(cov)

    result = run_cross_sectional_backtest(
        returns,
        weight_model,
        lookback=args.lookback,
        rebalance_frequency=args.rebalance,
        transaction_cost_bps=args.tcost_bps,
    )

    latest_weights = result.weights.iloc[-1].dropna()
    latest_weights = latest_weights[latest_weights > 0]

    mc_input = returns[latest_weights.index].tail(max(252, args.lookback))
    mc_risk = monte_carlo_var_cvar_from_history(
        mc_input,
        latest_weights / latest_weights.sum(),
        covariance_method="ledoit_wolf",
        n_sims=20_000,
        horizon=21,
        alpha=0.95,
        random_state=42,
    )

    print("=== Phase H1 End-to-End Summary ===")
    print(f"Universe size used: {returns.shape[1]} assets")
    print(f"Backtest period: {returns.index.min().date()} to {returns.index.max().date()}")

    for key, value in result.metrics.items():
        print(f"{key}: {value:.6f}")

    print("--- Monte Carlo (21-day horizon) ---")
    print(f"VaR@95% (loss): {mc_risk.var:.6f}")
    print(f"CVaR@95% (loss): {mc_risk.cvar:.6f}")
    print(f"Expected return: {mc_risk.expected_return:.6f}")


if __name__ == "__main__":
    main()
