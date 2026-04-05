"""hangar.homelab.results.store — Disk-based result storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class ResultStore:
    """Stores experiment results to disk.

    Structure:
        {base_dir}/{experiment_name}/
            config.json
            metrics.json
            prices.csv
            returns.csv
    """

    def __init__(self, base_dir: str | Path = "results") -> None:
        self.base_dir = Path(base_dir)

    def save(self, name: str, config: Dict[str, Any], metrics: Dict[str, Any],
             prices=None, returns=None) -> Path:
        """Save experiment results to disk."""
        exp_dir = self.base_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        with open(exp_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, cls=_NumpyEncoder)

        with open(exp_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=_NumpyEncoder)

        if prices is not None:
            prices.to_csv(exp_dir / "prices.csv")

        if returns is not None:
            returns.to_csv(exp_dir / "returns.csv")

        return exp_dir

    def load_metrics(self, name: str) -> Dict[str, Any]:
        """Load metrics for an experiment."""
        path = self.base_dir / name / "metrics.json"
        return json.loads(path.read_text())

    def list_experiments(self) -> list[str]:
        """List all saved experiments."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )
