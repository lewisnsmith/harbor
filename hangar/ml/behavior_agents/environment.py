"""hangar.ml.behavior_agents.environment — Gymnasium portfolio environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class EnvConfig:
    """Configuration for the portfolio environment."""

    obs_window: int = 60
    initial_cash: float = 1.0
    transaction_cost_bps: float = 5.0
    risk_free_rate: float = 0.0
    action_mode: str = "target"  # "target" or "delta"
    max_episode_steps: int = 252
    reward_risk_aversion: float = 1.0
    reward_turnover_penalty: float = 0.01


class PortfolioEnv(gym.Env):
    """Gymnasium environment for portfolio allocation.

    Observation space:
        Dict with keys:

        - ``returns_window``: ``Box(obs_window, n_assets)`` — trailing returns
        - ``current_weights``: ``Box(n_assets,)`` — current portfolio weights
        - ``rolling_vol``: ``Box(n_assets,)`` — 21-day rolling vol per asset
        - ``drawdown``: ``Box(1,)`` — current portfolio drawdown from peak

    Action space:
        If ``action_mode == "target"``:
            ``Box(n_assets,)`` in ``[0, 1]`` — target weights (normalized internally)
        If ``action_mode == "delta"``:
            ``Box(n_assets,)`` in ``[-1, 1]`` — weight changes

    Reward:
        ``r_p - lambda * r_p^2 - lambda_turnover * turnover``

    Parameters
    ----------
    returns
        Daily return panel (DatetimeIndex x tickers).
    config
        Environment configuration.
    reward_shaper
        Optional :class:`CompositeRewardShaper` for behavioral biases.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[EnvConfig] = None,
        reward_shaper: Optional[Any] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = EnvConfig()
        self._config = config
        self._reward_shaper = reward_shaper

        self._returns = returns.values.astype(np.float32)
        self._asset_names = list(returns.columns)
        self._dates = returns.index
        n = self.n_assets

        # Pre-compute rolling volatility (21-day)
        rv = returns.rolling(21).std().fillna(0.0).values.astype(np.float32)
        self._rolling_vol = rv

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "returns_window": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(config.obs_window, n), dtype=np.float32,
                ),
                "current_weights": spaces.Box(
                    low=0.0, high=1.0, shape=(n,), dtype=np.float32,
                ),
                "rolling_vol": spaces.Box(
                    low=0.0, high=np.inf, shape=(n,), dtype=np.float32,
                ),
                "drawdown": spaces.Box(
                    low=-1.0, high=0.0, shape=(1,), dtype=np.float32,
                ),
            }
        )

        # Action space
        if config.action_mode == "target":
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(n,), dtype=np.float32,
            )
        elif config.action_mode == "delta":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n,), dtype=np.float32,
            )
        else:
            raise ValueError(
                f"action_mode must be 'target' or 'delta', got {config.action_mode!r}"
            )

        # State variables (set in reset)
        self._current_step: int = 0
        self._episode_start: int = 0
        self._weights: np.ndarray = np.zeros(n, dtype=np.float32)
        self._portfolio_value: float = config.initial_cash
        self._peak_value: float = config.initial_cash
        self._steps_taken: int = 0

    @property
    def n_assets(self) -> int:
        return len(self._asset_names)

    @property
    def asset_names(self) -> List[str]:
        return self._asset_names

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        cfg = self._config
        n_total = len(self._returns)

        # Need obs_window for history + max_episode_steps for episode
        min_start = cfg.obs_window
        max_start = n_total - cfg.max_episode_steps
        if max_start <= min_start:
            max_start = min_start + 1

        self._episode_start = self.np_random.integers(min_start, max_start)
        self._current_step = self._episode_start
        self._weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self._portfolio_value = cfg.initial_cash
        self._peak_value = cfg.initial_cash
        self._steps_taken = 0

        obs = self._compute_observation()
        info: Dict[str, Any] = {"date": str(self._dates[self._current_step])}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        prev_weights = self._weights.copy()
        new_weights = self._normalize_action(action)

        # Turnover and transaction costs
        turnover = float(np.sum(np.abs(new_weights - prev_weights)))
        tc = turnover * self._config.transaction_cost_bps / 10_000.0

        # Apply new weights
        self._weights = new_weights

        # Portfolio return for this step
        asset_returns = self._returns[self._current_step]
        portfolio_return = float(np.dot(self._weights, asset_returns)) - tc

        # Update portfolio value and peak
        self._portfolio_value *= 1.0 + portfolio_return
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # Compute reward
        reward = self._compute_reward(portfolio_return, turnover)

        # Add behavioral reward shaping if configured
        if self._reward_shaper is not None:
            obs_win = self._config.obs_window
            start = max(0, self._current_step - obs_win)
            returns_history = self._returns[start : self._current_step]
            reward += self._reward_shaper.compute(
                portfolio_return,
                self._weights,
                prev_weights,
                returns_history,
                self._portfolio_value,
                self._peak_value,
            )

        # Advance step
        self._current_step += 1
        self._steps_taken += 1

        terminated = False
        truncated = self._steps_taken >= self._config.max_episode_steps
        if self._current_step >= len(self._returns):
            terminated = True

        obs = self._compute_observation() if not terminated else self._empty_obs()

        info: Dict[str, Any] = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "transaction_cost": tc,
            "portfolio_value": self._portfolio_value,
        }

        return obs, reward, terminated, truncated, info

    def _compute_observation(self) -> Dict[str, np.ndarray]:
        cfg = self._config
        step = self._current_step
        start = max(0, step - cfg.obs_window)

        # Returns window, padded with zeros if not enough history
        window = self._returns[start:step]
        if len(window) < cfg.obs_window:
            pad = np.zeros(
                (cfg.obs_window - len(window), self.n_assets), dtype=np.float32
            )
            window = np.concatenate([pad, window], axis=0)

        # Rolling vol at current step
        rv = self._rolling_vol[min(step, len(self._rolling_vol) - 1)]

        # Drawdown
        dd = (self._portfolio_value / self._peak_value) - 1.0

        return {
            "returns_window": window.astype(np.float32),
            "current_weights": self._weights.copy(),
            "rolling_vol": rv.copy(),
            "drawdown": np.array([dd], dtype=np.float32),
        }

    def _empty_obs(self) -> Dict[str, np.ndarray]:
        n = self.n_assets
        cfg = self._config
        return {
            "returns_window": np.zeros((cfg.obs_window, n), dtype=np.float32),
            "current_weights": self._weights.copy(),
            "rolling_vol": np.zeros(n, dtype=np.float32),
            "drawdown": np.array([0.0], dtype=np.float32),
        }

    def _compute_reward(self, portfolio_return: float, turnover: float) -> float:
        lam = self._config.reward_risk_aversion
        lam_turn = self._config.reward_turnover_penalty
        return portfolio_return - lam * portfolio_return**2 - lam_turn * turnover

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)

        if self._config.action_mode == "delta":
            raw = self._weights + action
        else:
            raw = action

        # Clip to non-negative (long-only)
        raw = np.maximum(raw, 0.0)

        total = raw.sum()
        if total > 0:
            return raw / total
        return np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
