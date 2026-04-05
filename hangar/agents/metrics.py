"""hangar.agents.metrics — Simulation analysis metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from hangar.ml.behavior_agents.multi_agent import compute_weight_similarity
from hangar.risk.regime_detection import detect_vol_shocks


def compute_crowding_index(
    agent_weights: Dict[str, pd.DataFrame],
    *,
    method: str = "cosine",
) -> pd.Series:
    """Average pairwise weight similarity over time.

    Delegates to ``hangar.ml.behavior_agents.multi_agent.compute_weight_similarity``.
    """
    return compute_weight_similarity(agent_weights, method=method)


def compute_flow_imbalance(orders: pd.DataFrame) -> pd.Series:
    """Net aggregate order flow magnitude per step.

    Returns a Series of the L1-norm of net flow at each time step.
    """
    return orders.abs().sum(axis=1).rename("flow_imbalance")


def compute_regime_labels(
    returns: pd.DataFrame,
    *,
    threshold_pct: float = 0.95,
    vol_window: int = 21,
) -> pd.Series:
    """Regime labels using volatility shock detection.

    Delegates to ``hangar.risk.regime_detection.detect_vol_shocks``.
    """
    market_returns = returns.mean(axis=1)
    return detect_vol_shocks(
        market_returns,
        threshold_pct=threshold_pct,
        vol_window=vol_window,
    )


def compute_return_autocorrelation(
    returns: pd.DataFrame,
    *,
    lag: int = 1,
    window: int = 63,
) -> pd.Series:
    """Rolling autocorrelation of market returns."""
    market_returns = returns.mean(axis=1)
    return market_returns.rolling(window).apply(
        lambda x: x.autocorr(lag=lag), raw=False
    ).rename("return_autocorr")


def compute_simulation_summary(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    agent_weights: Dict[str, pd.DataFrame],
    orders: pd.DataFrame,
) -> Dict[str, Any]:
    """Aggregate summary metrics for a simulation run."""
    market_returns = returns.mean(axis=1)

    # Annualized volatility
    vol = float(market_returns.std() * np.sqrt(252))

    # Return autocorrelation at lag 1
    autocorr = float(market_returns.autocorr(lag=1)) if len(market_returns) > 1 else 0.0

    # Crowding
    if agent_weights:
        crowding = compute_crowding_index(agent_weights)
        crowding_mean = float(crowding.mean())
        crowding_std = float(crowding.std())
    else:
        crowding_mean = 0.0
        crowding_std = 0.0

    # Flow imbalance
    flow = compute_flow_imbalance(orders)
    flow_mean = float(flow.mean())

    # Regime count (vol shocks)
    try:
        shocks = compute_regime_labels(returns)
        regime_count = int(shocks.sum())
    except (ValueError, Exception):
        regime_count = 0

    return {
        "annualized_vol": vol,
        "return_autocorrelation": autocorr,
        "crowding_mean": crowding_mean,
        "crowding_std": crowding_std,
        "flow_imbalance_mean": flow_mean,
        "regime_shock_count": regime_count,
        "n_agents": len(agent_weights),
        "n_steps": len(returns),
    }
