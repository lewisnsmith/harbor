"""
harbor.backtest.engine — Cross-sectional backtest engine.

Provides a rolling-window backtest loop that applies a user-supplied weight
function on each rebalance date, computes portfolio returns net of
transaction costs, and aggregates standard performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import pandas as pd

from harbor.backtest.metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    weights : pd.DataFrame
        Portfolio weights on each rebalance date (assets as columns).
    portfolio_returns : pd.Series
        Daily portfolio returns net of transaction costs.
    metrics : dict
        Summary statistics: annualized_return, annualized_volatility,
        sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio.
    """

    weights: pd.DataFrame
    portfolio_returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


def run_cross_sectional_backtest(
    returns: pd.DataFrame,
    weight_func: Callable[[pd.DataFrame, pd.Series], pd.Series],
    lookback: int = 126,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Run a rolling-window cross-sectional backtest.

    On each rebalance date the ``weight_func`` receives the trailing
    ``lookback``-day return window and the current portfolio weights, and
    returns target weights.  Between rebalances, weights drift with asset
    returns.  Transaction costs are deducted proportionally to weight
    turnover on rebalance dates.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns (dates x tickers).  Must have at least
        ``lookback + rebalance_frequency`` rows.
    weight_func : callable
        ``(window: pd.DataFrame, current_weights: pd.Series) -> pd.Series``
        Maps a lookback window and current weights to target weights.
        Returned weights should sum to approximately 1.
    lookback : int
        Number of trailing trading days passed to ``weight_func``.
    rebalance_frequency : int
        Trading days between rebalances.
    transaction_cost_bps : float
        One-way transaction cost in basis points (applied to absolute
        weight changes on each rebalance).

    Returns
    -------
    BacktestResult
        Weights history, daily portfolio returns, and summary metrics.

    Raises
    ------
    ValueError
        If ``returns`` has insufficient rows for the lookback window.
    """
    n_rows, n_assets = returns.shape
    if n_rows <= lookback:
        raise ValueError(
            f"returns has {n_rows} rows but lookback requires at least {lookback + 1}."
        )

    tc_rate = transaction_cost_bps / 10_000.0
    dates = returns.index
    assets = returns.columns

    # Pre-allocate outputs
    weight_records: list[tuple[pd.Timestamp, pd.Series]] = []
    port_returns = pd.Series(0.0, index=dates, dtype=float)

    current_weights = pd.Series(0.0, index=assets, dtype=float)
    days_since_rebalance = rebalance_frequency  # force rebalance on first eligible day

    for i in range(lookback, n_rows):
        date = dates[i]
        day_returns = returns.iloc[i]

        # Rebalance check
        if days_since_rebalance >= rebalance_frequency:
            window = returns.iloc[i - lookback : i]
            target_weights = weight_func(window, current_weights)

            # Align to asset universe
            target_weights = target_weights.reindex(assets, fill_value=0.0)

            # Normalize
            w_sum = target_weights.sum()
            if w_sum > 0:
                target_weights = target_weights / w_sum

            # Transaction cost: proportional to turnover
            turnover = (target_weights - current_weights).abs().sum()
            tc = turnover * tc_rate

            current_weights = target_weights.copy()
            weight_records.append((date, current_weights.copy()))
            days_since_rebalance = 0
        else:
            tc = 0.0

        # Portfolio return for this day
        port_ret = (current_weights * day_returns).sum() - tc
        port_returns.iloc[i] = port_ret

        # Drift weights with asset returns
        grown = current_weights * (1 + day_returns)
        total = grown.sum()
        if total > 0:
            current_weights = grown / total
        days_since_rebalance += 1

    # Trim to active period
    active_returns = port_returns.iloc[lookback:]

    # Build weights DataFrame
    if weight_records:
        w_dates, w_series = zip(*weight_records)
        weights_df = pd.DataFrame(list(w_series), index=pd.DatetimeIndex(w_dates))
    else:
        weights_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=assets)

    # Compute summary metrics
    metrics = _compute_metrics(active_returns)

    return BacktestResult(
        weights=weights_df,
        portfolio_returns=active_returns,
        metrics=metrics,
    )


def _compute_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute standard performance metrics from a return series."""
    arr = returns.values
    arr_clean = arr[np.isfinite(arr)]

    if len(arr_clean) < 2:
        return {
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "calmar_ratio": float("nan"),
        }

    ann_ret = float(np.mean(arr_clean) * 252)
    ann_vol = float(np.std(arr_clean, ddof=1) * np.sqrt(252))

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
    }
