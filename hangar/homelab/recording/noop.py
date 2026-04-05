"""hangar.homelab.recording.noop — No-op recorder (default when tracing disabled)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from hangar.homelab.venue.protocol import VenueSnapshot


class NoopRecorder:
    """Recorder that discards all data. Used when tracing is disabled."""

    def start_experiment(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        pass

    def record_step(
        self,
        step: int,
        snapshot: VenueSnapshot,
        orders: Dict[str, np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
        pass

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        pass

    def end_experiment(self, summary: Dict[str, Any]) -> None:
        pass

    def flush(self) -> None:
        pass
