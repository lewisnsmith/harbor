"""hangar.homelab.venue.protocol — Venue protocol and VenueSnapshot dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VenueSnapshot:
    """Normalized observation that every agent receives from a venue.

    All venue types (equity, prediction market, etc.) produce snapshots
    in this common format so agents can be venue-agnostic.
    """

    timestamp: pd.Timestamp
    step: int
    assets: List[str]
    prices: np.ndarray           # (n_assets,)
    returns: np.ndarray          # (n_assets,)
    volume: np.ndarray           # (n_assets,)
    spread: np.ndarray           # (n_assets,)
    returns_history: np.ndarray  # (lookback, n_assets)
    market_type: str             # "equity", "prediction", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return len(self.assets)


@runtime_checkable
class Venue(Protocol):
    """Protocol for market venue implementations.

    A venue manages price formation, order matching, and state evolution.
    Implementations wrap specific market simulators (equity, prediction, etc.)
    and expose them through the normalized VenueSnapshot interface.
    """

    def reset(self, seed: int) -> VenueSnapshot:
        """Reset the venue to initial state with the given seed."""
        ...

    def step(self, orders: Dict[str, np.ndarray]) -> VenueSnapshot:
        """Advance one step given agent orders. Returns new snapshot."""
        ...

    @property
    def config(self) -> Dict[str, Any]:
        """Return venue configuration as a serializable dict."""
        ...
