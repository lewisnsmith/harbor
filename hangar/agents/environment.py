"""harbor.agents.environment — Stepped market simulation with price impact."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketConfig:
    """Configuration for the synthetic market environment."""

    n_steps: int = 500
    n_assets: int = 10
    base_volatility: float = 0.01
    base_drift: float = 0.0002
    temporary_impact: float = 0.1
    permanent_impact: float = 0.01
    correlation: float = 0.3
    lookback_window: int = 60
    seed: int = 42


@dataclass
class MarketState:
    """Observable market state at a single time step."""

    prices: pd.Series
    returns: pd.Series
    returns_history: pd.DataFrame
    step: int
    date: pd.Timestamp


class MarketEnvironment:
    """Stepped market simulator with agent-driven price impact.

    Price formation per step:
    1. Sample base returns from multivariate normal (drift + vol + correlation).
    2. Aggregate net order flow from all agents.
    3. Temporary impact: ``r_impact = -temporary_impact * net_flow``.
    4. Permanent impact: ``price_shift = -permanent_impact * sign(flow) * sqrt(|flow|)``.
    5. Final return = base + temporary + permanent.
    """

    def __init__(self, config: Optional[MarketConfig] = None) -> None:
        self.config = config or MarketConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Build correlation matrix
        n = self.config.n_assets
        corr = np.full((n, n), self.config.correlation)
        np.fill_diagonal(corr, 1.0)
        vols = np.full(n, self.config.base_volatility)
        self._cov = np.outer(vols, vols) * corr
        self._drift = np.full(n, self.config.base_drift)
        self._asset_names = [f"asset_{i}" for i in range(n)]

        self._prices: Optional[np.ndarray] = None
        self._returns_buffer: list = []
        self._step: int = 0
        self._dates: Optional[pd.DatetimeIndex] = None

    def reset(self) -> MarketState:
        """Reset environment and return the initial state."""
        self._rng = np.random.default_rng(self.config.seed)
        self._prices = np.ones(self.config.n_assets) * 100.0
        self._returns_buffer = []
        self._step = 0
        self._dates = pd.bdate_range("2020-01-01", periods=self.config.n_steps + 1)

        initial_returns = np.zeros(self.config.n_assets)
        self._returns_buffer.append(initial_returns)
        return self._make_state(initial_returns)

    def step(self, agent_orders: Dict[str, np.ndarray]) -> MarketState:
        """Advance one step given agent order arrays.

        Parameters
        ----------
        agent_orders
            Dict mapping agent name to order array (target_weight deltas).
            Each array has shape ``(n_assets,)``.
        """
        if self._prices is None:
            raise RuntimeError("Call reset() before step().")

        self._step += 1
        cfg = self.config

        # 1. Base returns from multivariate normal
        base_returns = self._rng.multivariate_normal(self._drift, self._cov)

        # 2. Aggregate net order flow
        if agent_orders:
            net_flow = np.sum(list(agent_orders.values()), axis=0)
        else:
            net_flow = np.zeros(cfg.n_assets)

        # 3. Temporary impact
        temp_impact = -cfg.temporary_impact * net_flow

        # 4. Permanent impact
        perm_impact = -cfg.permanent_impact * np.sign(net_flow) * np.sqrt(
            np.abs(net_flow)
        )

        # 5. Final returns
        final_returns = base_returns + temp_impact + perm_impact

        # Update prices
        self._prices = self._prices * (1.0 + final_returns)
        self._returns_buffer.append(final_returns)

        return self._make_state(final_returns)

    def _make_state(self, latest_returns: np.ndarray) -> MarketState:
        prices = pd.Series(self._prices, index=self._asset_names, name="prices")
        returns = pd.Series(latest_returns, index=self._asset_names, name="returns")

        # Build returns history (up to lookback_window)
        window = self.config.lookback_window
        buf = self._returns_buffer[-window:]
        history = pd.DataFrame(buf, columns=self._asset_names)

        return MarketState(
            prices=prices,
            returns=returns,
            returns_history=history,
            step=self._step,
            date=self._dates[self._step],
        )
