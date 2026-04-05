"""hangar.agents.autonomous_agent — Personality-parameterized autonomous agents."""

from __future__ import annotations

import numpy as np

from hangar.agents.base_agent import AgentConfig, BaseAgent
from hangar.agents.environment import MarketState


class AutonomousAgent(BaseAgent):
    """Autonomous agent with 2D personality (risk_appetite, reactivity).

    Blends momentum and mean-reversion signals weighted by reactivity,
    scales positions by volatility target derived from risk_appetite,
    and applies rebalance damping proportional to reactivity.

    Parameters
    ----------
    risk_appetite : float
        Controls volatility target. Range [0.1, 1.0].
        target_vol = 0.01 + 0.09 * risk_appetite (1% to 10% annualized).
    reactivity : float
        Controls signal blend and rebalance speed. Range [0.1, 1.0].
        lookback = round(5 + 55 * (1 - reactivity)) (5 to 55 steps).
        Higher reactivity = more momentum, faster rebalancing.
    """

    def __init__(
        self,
        config: AgentConfig,
        n_assets: int,
        risk_appetite: float = 0.5,
        reactivity: float = 0.5,
    ) -> None:
        super().__init__(config, n_assets)
        self.risk_appetite = float(risk_appetite)
        self.reactivity = float(reactivity)
        self.lookback = round(5 + 55 * (1 - self.reactivity))
        self.target_vol = 0.01 + 0.09 * self.risk_appetite

    def observe(self, state: MarketState) -> None:
        self._state = state

    def decide(self) -> np.ndarray:
        state = self._state
        history = state.returns_history
        window = history.iloc[-min(self.lookback, len(history)) :]

        if len(window) < 2:
            return np.full(self.n_assets, 1.0 / self.n_assets)

        # --- Momentum signal: cumulative return per asset ---
        s_mom = (1.0 + window).prod().values - 1.0

        # --- Mean-reversion signal: z-score of price vs rolling mean ---
        # Reconstruct price path from returns (arbitrary base = 1.0)
        cum_returns = (1.0 + window).cumprod()
        prices_reconstructed = cum_returns.values  # shape (T, n_assets)
        rolling_len = min(len(prices_reconstructed), self.lookback)
        price_mean = prices_reconstructed[-rolling_len:].mean(axis=0)
        price_std = prices_reconstructed[-rolling_len:].std(axis=0)
        current_price = prices_reconstructed[-1]
        # Negative z-score: below mean = buy signal (mean reversion)
        safe_std = np.where(price_std > 1e-8, price_std, 1.0)
        s_mr = -(current_price - price_mean) / safe_std
        s_mr = np.clip(s_mr, -3.0, 3.0)

        # --- Z-score normalize each signal ---
        s_mom = self._znorm(s_mom)
        s_mr = self._znorm(s_mr)

        # --- Blend by reactivity ---
        raw = self.reactivity * s_mom + (1.0 - self.reactivity) * s_mr

        # --- Vol scaling (non-directional) ---
        realized_vol = window.std().values * np.sqrt(252)
        vol_scale = np.clip(
            self.target_vol / (realized_vol + 1e-8), 0.0, 10.0
        )
        scaled = raw * vol_scale

        # --- Long-only normalization: shift then normalize ---
        shifted = scaled - scaled.min() + 1e-8
        target = shifted / shifted.sum()

        return target

    def act(self, state: MarketState) -> np.ndarray:
        """Observe, decide, apply rebalance damping, return orders."""
        self.observe(state)
        target = self.decide()

        # Damped rebalance: low reactivity moves slowly toward target
        actual = self._weights + self.reactivity * (target - self._weights)

        # Renormalize after damping (maintains sum-to-one)
        total = actual.sum()
        if total > 0:
            actual = actual / total
        else:
            actual = np.full(self.n_assets, 1.0 / self.n_assets)

        orders = actual - self._weights
        self._weights = actual.copy()
        return orders

    @staticmethod
    def _znorm(s: np.ndarray) -> np.ndarray:
        """Z-score normalize; skip if std is near zero."""
        std = s.std()
        if std < 1e-8:
            return s
        return (s - s.mean()) / std


class RandomAgent(BaseAgent):
    """Null-model agent: uniform random weights each step.

    Used as the control condition in coordination experiments.
    """

    def __init__(
        self,
        config: AgentConfig,
        n_assets: int,
        seed: int = 0,
    ) -> None:
        super().__init__(config, n_assets)
        self._rng = np.random.default_rng(seed)

    def observe(self, state: MarketState) -> None:
        self._state = state

    def decide(self) -> np.ndarray:
        raw = self._rng.uniform(0, 1, self.n_assets)
        return raw / raw.sum()
