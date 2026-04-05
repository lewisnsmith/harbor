"""hangar.homelab.recording.protocol — Recorder protocol for trace recording.

Flight integration: when Flight is ready, implement a FlightRecorder
that satisfies this protocol. The runner doesn't need to change.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np

from hangar.homelab.venue.protocol import VenueSnapshot


@runtime_checkable
class Recorder(Protocol):
    """Protocol for experiment trace recording."""

    def start_experiment(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Called once at the start of an experiment run."""
        ...

    def record_step(
        self,
        step: int,
        snapshot: VenueSnapshot,
        orders: Dict[str, np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
        """Called after each simulation step."""
        ...

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record an arbitrary event (agent action, error, etc.)."""
        ...

    def end_experiment(self, summary: Dict[str, Any]) -> None:
        """Called once at the end of an experiment run."""
        ...

    def flush(self) -> None:
        """Flush any buffered data to storage."""
        ...
