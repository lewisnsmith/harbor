"""harbor.agents.simulation — Simulation runner for multi-agent market."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from harbor.agents.base_agent import BaseAgent
from harbor.agents.environment import MarketEnvironment


@dataclass
class SimulationResult:
    """Output of a multi-agent simulation run."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    agent_weights: Dict[str, pd.DataFrame]
    agent_returns: Dict[str, pd.Series]
    orders: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)


def run_simulation(
    market: MarketEnvironment,
    agents: List[BaseAgent],
) -> SimulationResult:
    """Run the full simulation loop.

    For each step:
    1. Each agent calls ``act(state)`` → orders (weight deltas).
    2. Orders are aggregated.
    3. ``market.step(orders)`` → new state.
    4. Prices, weights, and returns are recorded.
    """
    cfg = market.config
    state = market.reset()
    n_assets = cfg.n_assets
    n_steps = cfg.n_steps
    asset_names = state.prices.index.tolist()

    # Pre-allocate storage
    price_records = np.zeros((n_steps + 1, n_assets))
    return_records = np.zeros((n_steps + 1, n_assets))
    order_records = np.zeros((n_steps, n_assets))
    weight_records = {a.name: np.zeros((n_steps + 1, n_assets)) for a in agents}
    ret_records = {a.name: np.zeros(n_steps) for a in agents}

    # Record initial state
    price_records[0] = state.prices.values
    return_records[0] = 0.0
    for a in agents:
        weight_records[a.name][0] = a.current_weights

    # Simulation loop
    for t in range(n_steps):
        agent_orders = {}
        for a in agents:
            orders = a.act(state)
            agent_orders[a.name] = orders

        # Aggregate orders for impact model
        state = market.step(agent_orders)

        # Record state
        price_records[t + 1] = state.prices.values
        return_records[t + 1] = state.returns.values
        order_records[t] = np.sum(list(agent_orders.values()), axis=0)

        # Per-agent tracking
        for a in agents:
            weight_records[a.name][t + 1] = a.current_weights
            # Portfolio return for this step (weights * asset returns)
            port_ret = float(np.dot(a.current_weights, state.returns.values))
            # Transaction cost
            tc = float(np.sum(np.abs(agent_orders[a.name]))) * (
                a.config.transaction_cost_bps / 10_000.0
            )
            ret_records[a.name][t] = port_ret - tc

    # Build DataFrames
    dates = pd.bdate_range("2020-01-01", periods=n_steps + 1)
    prices = pd.DataFrame(price_records, index=dates, columns=asset_names)
    returns = pd.DataFrame(return_records, index=dates, columns=asset_names)
    orders = pd.DataFrame(order_records, index=dates[1:], columns=asset_names)

    agent_weights = {
        name: pd.DataFrame(w, index=dates, columns=asset_names)
        for name, w in weight_records.items()
    }
    agent_returns = {
        name: pd.Series(r, index=dates[1:], name=name)
        for name, r in ret_records.items()
    }

    return SimulationResult(
        prices=prices,
        returns=returns,
        agent_weights=agent_weights,
        agent_returns=agent_returns,
        orders=orders,
    )
