"""harbor.agents.rule_agents — Rule-based trading agents."""

from __future__ import annotations

import numpy as np

from harbor.agents.base_agent import AgentConfig, BaseAgent
from harbor.agents.environment import MarketState


class MomentumAgent(BaseAgent):
    """Buys recent winners, sells recent losers.

    Ranks assets by trailing cumulative return over ``lookback`` steps,
    then overweights the top half and underweights the bottom half.
    """

    def __init__(
        self,
        config: AgentConfig,
        n_assets: int,
        lookback: int = 21,
    ) -> None:
        super().__init__(config, n_assets)
        self.lookback = lookback

    def observe(self, state: MarketState) -> None:
        self._state = state

    def decide(self) -> np.ndarray:
        history = self._state.returns_history
        if len(history) < 2:
            return np.full(self.n_assets, 1.0 / self.n_assets)

        window = history.iloc[-min(self.lookback, len(history)) :]
        cum_returns = (1.0 + window).prod() - 1.0

        # Rank: higher return → higher weight
        ranks = cum_returns.values.argsort().argsort().astype(float) + 1.0
        weights = ranks / ranks.sum()
        return weights


class MeanReversionAgent(BaseAgent):
    """Contrarian agent: buys recent losers, sells recent winners.

    Inverse of MomentumAgent — overweights assets that have
    underperformed over the lookback window.
    """

    def __init__(
        self,
        config: AgentConfig,
        n_assets: int,
        lookback: int = 21,
    ) -> None:
        super().__init__(config, n_assets)
        self.lookback = lookback

    def observe(self, state: MarketState) -> None:
        self._state = state

    def decide(self) -> np.ndarray:
        history = self._state.returns_history
        if len(history) < 2:
            return np.full(self.n_assets, 1.0 / self.n_assets)

        window = history.iloc[-min(self.lookback, len(history)) :]
        cum_returns = (1.0 + window).prod() - 1.0

        # Inverse rank: lower return → higher weight
        ranks = (-cum_returns.values).argsort().argsort().astype(float) + 1.0
        weights = ranks / ranks.sum()
        return weights


class VolTargetAgent(BaseAgent):
    """Scales exposure inversely with recent realized volatility.

    Each asset's weight is proportional to
    ``target_vol / realized_vol_i``, then normalized to sum to 1.
    """

    def __init__(
        self,
        config: AgentConfig,
        n_assets: int,
        target_vol: float = 0.10,
        vol_window: int = 21,
    ) -> None:
        super().__init__(config, n_assets)
        self.target_vol = target_vol
        self.vol_window = vol_window

    def observe(self, state: MarketState) -> None:
        self._state = state

    def decide(self) -> np.ndarray:
        history = self._state.returns_history
        if len(history) < 3:
            return np.full(self.n_assets, 1.0 / self.n_assets)

        window = history.iloc[-min(self.vol_window, len(history)) :]
        realized_vol = window.std().values

        # Annualize for comparison with target
        annualized_vol = realized_vol * np.sqrt(252)

        # Inverse vol weighting
        inv_vol = np.where(
            annualized_vol > 1e-8,
            self.target_vol / annualized_vol,
            0.0,
        )

        total = inv_vol.sum()
        if total > 0:
            weights = inv_vol / total
        else:
            weights = np.full(self.n_assets, 1.0 / self.n_assets)

        return weights
