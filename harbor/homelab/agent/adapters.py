"""harbor.homelab.agent.adapters — Bridge existing BaseAgent to homelab protocols."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harbor.agents.base_agent import BaseAgent
from harbor.agents.environment import MarketState
from harbor.homelab.venue.protocol import VenueSnapshot


class LegacyAgentAdapter:
    """Wraps an existing BaseAgent so it satisfies the Observable protocol.

    Converts VenueSnapshot → MarketState for the inner agent's observe/decide
    cycle. The existing rule agents (Momentum, MeanReversion, VolTarget)
    work unchanged through this adapter.
    """

    def __init__(self, inner: BaseAgent) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        return self._inner.name

    def observe(self, snapshot: VenueSnapshot) -> None:
        state = self._snapshot_to_state(snapshot)
        self._inner.observe(state)

    def decide(self) -> np.ndarray:
        return self._inner.decide()

    def act(self, snapshot: VenueSnapshot) -> np.ndarray:
        state = self._snapshot_to_state(snapshot)
        return self._inner.act(state)

    @property
    def current_weights(self) -> np.ndarray:
        return self._inner.current_weights

    @staticmethod
    def _snapshot_to_state(snapshot: VenueSnapshot) -> MarketState:
        """Convert VenueSnapshot to the MarketState the legacy agent expects."""
        return MarketState(
            prices=pd.Series(snapshot.prices, index=snapshot.assets, name="prices"),
            returns=pd.Series(snapshot.returns, index=snapshot.assets, name="returns"),
            returns_history=pd.DataFrame(
                snapshot.returns_history, columns=snapshot.assets
            ),
            step=snapshot.step,
            date=snapshot.timestamp,
        )
