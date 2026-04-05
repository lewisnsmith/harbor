"""harbor.agents.base_agent — Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from harbor.agents.environment import MarketState


@dataclass
class AgentConfig:
    """Per-agent configuration."""

    name: str
    initial_cash: float = 1.0
    transaction_cost_bps: float = 5.0


class BaseAgent(ABC):
    """Abstract base class for simulation agents.

    Subclasses implement ``observe`` and ``decide``. The ``act`` method
    orchestrates the full cycle: observe → decide → compute orders.
    """

    def __init__(self, config: AgentConfig, n_assets: int) -> None:
        self.config = config
        self.n_assets = n_assets
        self._weights = np.zeros(n_assets)
        self._state: MarketState | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def current_weights(self) -> np.ndarray:
        return self._weights.copy()

    @current_weights.setter
    def current_weights(self, w: np.ndarray) -> None:
        self._weights = w.copy()

    @abstractmethod
    def observe(self, state: MarketState) -> None:
        """Receive the latest market state."""

    @abstractmethod
    def decide(self) -> np.ndarray:
        """Return target portfolio weights (n_assets,). Must sum to <= 1."""

    def act(self, state: MarketState) -> np.ndarray:
        """Full cycle: observe → decide → return order (weight deltas)."""
        self.observe(state)
        target_weights = self.decide()

        # Normalize if sum exceeds 1
        total = np.sum(np.abs(target_weights))
        if total > 1.0:
            target_weights = target_weights / total

        orders = target_weights - self._weights
        self._weights = target_weights.copy()
        return orders
