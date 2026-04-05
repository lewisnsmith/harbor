"""harbor.homelab.evaluation.registry — Metrics registry wrapping existing metric functions."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from harbor.agents.metrics import (
    compute_crowding_index,
    compute_flow_imbalance,
    compute_regime_labels,
    compute_return_autocorrelation,
)
from harbor.backtest.metrics import calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio


class MetricsRegistry:
    """Registry of named metric functions.

    Wraps existing metric functions from harbor.agents.metrics and
    harbor.backtest.metrics into a unified interface. Custom metrics
    can be registered at runtime.
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, Callable[..., Any]] = {}
        self._register_defaults()

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a metric function by name."""
        self._metrics[name] = fn

    def available(self) -> list[str]:
        """List registered metric names."""
        return sorted(self._metrics.keys())

    def compute(self, name: str, **kwargs: Any) -> Any:
        """Compute a single metric by name."""
        if name not in self._metrics:
            raise KeyError(
                f"Unknown metric: {name!r}. Available: {self.available()}"
            )
        return self._metrics[name](**kwargs)

    def compute_all(
        self,
        names: list[str],
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        agent_weights: Dict[str, pd.DataFrame],
        orders: pd.DataFrame,
        agent_returns: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        """Compute multiple metrics given simulation outputs."""
        results: Dict[str, Any] = {}
        market_returns = returns.mean(axis=1)

        for name in names:
            try:
                if name == "crowding_index":
                    val = compute_crowding_index(agent_weights)
                    results[name] = float(val.mean())
                elif name == "flow_imbalance":
                    val = compute_flow_imbalance(orders)
                    results[name] = float(val.mean())
                elif name == "regime_labels":
                    val = compute_regime_labels(returns)
                    results[name] = int(val.sum())
                elif name == "return_autocorrelation":
                    val = compute_return_autocorrelation(returns)
                    results[name] = float(val.dropna().mean())
                elif name == "annualized_vol":
                    results[name] = float(market_returns.std() * np.sqrt(252))
                elif name == "sharpe_ratio":
                    results[name] = sharpe_ratio(market_returns)
                elif name == "sortino_ratio":
                    results[name] = sortino_ratio(market_returns)
                elif name == "max_drawdown":
                    results[name] = max_drawdown(market_returns)
                elif name == "calmar_ratio":
                    results[name] = calmar_ratio(market_returns)
                elif name in self._metrics:
                    results[name] = self._metrics[name](
                        prices=prices,
                        returns=returns,
                        agent_weights=agent_weights,
                        orders=orders,
                    )
                else:
                    results[name] = f"ERROR: unknown metric {name!r}"
            except Exception as e:
                results[name] = f"ERROR: {e}"

        return results

    def _register_defaults(self) -> None:
        """Register built-in metrics."""
        for name in [
            "crowding_index",
            "flow_imbalance",
            "regime_labels",
            "return_autocorrelation",
            "annualized_vol",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ]:
            self._metrics[name] = lambda **kw: None  # placeholder for compute_all
