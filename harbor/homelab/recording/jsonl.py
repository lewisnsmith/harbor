"""harbor.homelab.recording.jsonl — JSONL trace recorder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from harbor.homelab.venue.protocol import VenueSnapshot


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class JsonlRecorder:
    """Writes one JSON line per event to a trace file.

    Output structure:
        {output_dir}/{experiment_id}/trace.jsonl
        {output_dir}/{experiment_id}/summary.json
    """

    def __init__(self, output_dir: str, experiment_id: str | None = None) -> None:
        self._output_dir = Path(output_dir)
        self._experiment_id = experiment_id or f"exp_{int(time.time())}"
        self._exp_dir = self._output_dir / self._experiment_id
        self._file = None

    def start_experiment(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        self._file = open(self._exp_dir / "trace.jsonl", "w")
        self._write_line({
            "type": "experiment_start",
            "config": config,
            "metadata": metadata,
            "timestamp": time.time(),
        })

    def record_step(
        self,
        step: int,
        snapshot: VenueSnapshot,
        orders: Dict[str, np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
        self._write_line({
            "type": "step",
            "step": step,
            "timestamp": str(snapshot.timestamp),
            "prices": snapshot.prices,
            "returns": snapshot.returns,
            "volume": snapshot.volume,
            "orders": {k: v for k, v in orders.items()},
            "metrics": metrics,
        })

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self._write_line({
            "type": "event",
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
        })

    def end_experiment(self, summary: Dict[str, Any]) -> None:
        self._write_line({
            "type": "experiment_end",
            "summary": summary,
            "timestamp": time.time(),
        })
        # Also write summary as standalone JSON
        with open(self._exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=_NumpyEncoder)
        self.flush()
        if self._file:
            self._file.close()
            self._file = None

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    @property
    def experiment_dir(self) -> Path:
        return self._exp_dir

    def _write_line(self, data: Dict[str, Any]) -> None:
        if self._file:
            self._file.write(json.dumps(data, cls=_NumpyEncoder) + "\n")
