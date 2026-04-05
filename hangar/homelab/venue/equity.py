"""hangar.homelab.venue.equity — EquityVenue adapter wrapping MarketEnvironment."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from hangar.agents.environment import MarketConfig, MarketEnvironment, MarketState
from hangar.homelab.venue.protocol import VenueSnapshot


class EquityVenue:
    """Adapter that wraps the existing MarketEnvironment as a Venue.

    Converts MarketState → VenueSnapshot, synthesizing volume and spread
    fields from order flow and volatility since the underlying environment
    does not produce them directly.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        params = params or {}
        self._market_config = MarketConfig(**params)
        self._env = MarketEnvironment(self._market_config)
        self._last_orders: Dict[str, np.ndarray] = {}
        self._cumulative_volume = np.zeros(self._market_config.n_assets)

    def reset(self, seed: int) -> VenueSnapshot:
        """Reset the venue with a new seed."""
        self._market_config = MarketConfig(
            n_steps=self._market_config.n_steps,
            n_assets=self._market_config.n_assets,
            base_volatility=self._market_config.base_volatility,
            base_drift=self._market_config.base_drift,
            temporary_impact=self._market_config.temporary_impact,
            permanent_impact=self._market_config.permanent_impact,
            correlation=self._market_config.correlation,
            lookback_window=self._market_config.lookback_window,
            seed=seed,
        )
        self._env = MarketEnvironment(self._market_config)
        self._last_orders = {}
        self._cumulative_volume = np.zeros(self._market_config.n_assets)
        state = self._env.reset()
        return self._to_snapshot(state)

    def step(self, orders: Dict[str, np.ndarray]) -> VenueSnapshot:
        """Advance one step given agent orders."""
        self._last_orders = orders
        state = self._env.step(orders)
        return self._to_snapshot(state)

    @property
    def config(self) -> Dict[str, Any]:
        """Return venue configuration as a serializable dict."""
        return {"type": "equity", "params": asdict(self._market_config)}

    def _to_snapshot(self, state: MarketState) -> VenueSnapshot:
        """Convert MarketState to VenueSnapshot."""
        n = self._market_config.n_assets
        assets = state.prices.index.tolist()

        # Synthesize volume from order flow
        if self._last_orders:
            step_volume = np.sum(
                [np.abs(o) for o in self._last_orders.values()], axis=0
            )
        else:
            step_volume = np.zeros(n)
        self._cumulative_volume += step_volume

        # Synthesize spread from base volatility (wider spread = higher vol)
        spread = np.full(n, self._market_config.base_volatility * 2.0)

        # Build returns history as numpy array
        returns_history = state.returns_history.values

        return VenueSnapshot(
            timestamp=state.date,
            step=state.step,
            assets=assets,
            prices=state.prices.values.copy(),
            returns=state.returns.values.copy(),
            volume=step_volume,
            spread=spread,
            returns_history=returns_history,
            market_type="equity",
            metadata={
                "fee_schedule": {
                    "temporary_impact": self._market_config.temporary_impact,
                    "permanent_impact": self._market_config.permanent_impact,
                },
                "cumulative_volume": self._cumulative_volume.copy(),
            },
        )
