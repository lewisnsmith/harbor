"""harbor.homelab.evaluation.summary — Evaluation pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from harbor.homelab.evaluation.registry import MetricsRegistry


def evaluate(
    metric_names: List[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    agent_weights: Dict[str, pd.DataFrame],
    orders: pd.DataFrame,
    agent_returns: Dict[str, pd.Series],
    registry: MetricsRegistry | None = None,
) -> Dict[str, Any]:
    """Run the evaluation pipeline: compute all requested metrics.

    Parameters
    ----------
    metric_names
        List of metric names to compute.
    prices, returns, agent_weights, orders, agent_returns
        Simulation outputs.
    registry
        Optional custom MetricsRegistry. Uses default if None.

    Returns
    -------
    dict
        Metric name → computed value.
    """
    if registry is None:
        registry = MetricsRegistry()

    return registry.compute_all(
        names=metric_names,
        prices=prices,
        returns=returns,
        agent_weights=agent_weights,
        orders=orders,
        agent_returns=agent_returns,
    )
