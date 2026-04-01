"""Tests for harbor.homelab.evaluation — MetricsRegistry and evaluate pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.homelab.evaluation.registry import MetricsRegistry
from harbor.homelab.evaluation.summary import evaluate


def _make_sim_outputs(n_steps: int = 100, n_assets: int = 3, n_agents: int = 2):
    """Build minimal simulation outputs for testing metrics."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_steps + 1)
    assets = [f"asset_{i}" for i in range(n_assets)]

    returns_data = rng.standard_normal((n_steps + 1, n_assets)) * 0.01
    prices_data = 100.0 * np.cumprod(1 + returns_data, axis=0)

    prices = pd.DataFrame(prices_data, index=dates, columns=assets)
    returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    orders = pd.DataFrame(
        rng.standard_normal((n_steps, n_assets)) * 0.05,
        index=dates[1:],
        columns=assets,
    )

    agent_weights = {}
    agent_returns = {}
    for i in range(n_agents):
        name = f"agent_{i}"
        w = np.abs(rng.standard_normal((n_steps + 1, n_assets)))
        w = w / w.sum(axis=1, keepdims=True)
        agent_weights[name] = pd.DataFrame(w, index=dates, columns=assets)
        agent_returns[name] = pd.Series(
            rng.standard_normal(n_steps) * 0.01, index=dates[1:], name=name
        )

    return prices, returns, agent_weights, orders, agent_returns


class TestMetricsRegistry:
    def test_available_metrics(self):
        reg = MetricsRegistry()
        avail = reg.available()
        assert "sharpe_ratio" in avail
        assert "crowding_index" in avail
        assert "annualized_vol" in avail

    def test_compute_all_basic(self):
        prices, returns, agent_weights, orders, agent_returns = _make_sim_outputs()
        reg = MetricsRegistry()
        results = reg.compute_all(
            names=["annualized_vol", "sharpe_ratio", "max_drawdown"],
            prices=prices,
            returns=returns,
            agent_weights=agent_weights,
            orders=orders,
            agent_returns=agent_returns,
        )
        assert "annualized_vol" in results
        assert isinstance(results["annualized_vol"], float)
        assert results["annualized_vol"] > 0

    def test_compute_all_crowding(self):
        prices, returns, agent_weights, orders, agent_returns = _make_sim_outputs()
        reg = MetricsRegistry()
        results = reg.compute_all(
            names=["crowding_index", "flow_imbalance"],
            prices=prices,
            returns=returns,
            agent_weights=agent_weights,
            orders=orders,
            agent_returns=agent_returns,
        )
        assert "crowding_index" in results
        assert "flow_imbalance" in results

    def test_unknown_metric_returns_error(self):
        prices, returns, agent_weights, orders, agent_returns = _make_sim_outputs()
        reg = MetricsRegistry()
        results = reg.compute_all(
            names=["nonexistent_metric"],
            prices=prices,
            returns=returns,
            agent_weights=agent_weights,
            orders=orders,
            agent_returns=agent_returns,
        )
        assert "ERROR" in str(results["nonexistent_metric"])


class TestEvaluate:
    def test_evaluate_pipeline(self):
        prices, returns, agent_weights, orders, agent_returns = _make_sim_outputs()
        results = evaluate(
            metric_names=["annualized_vol", "sharpe_ratio"],
            prices=prices,
            returns=returns,
            agent_weights=agent_weights,
            orders=orders,
            agent_returns=agent_returns,
        )
        assert "annualized_vol" in results
        assert "sharpe_ratio" in results
