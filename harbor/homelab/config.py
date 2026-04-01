"""harbor.homelab.config — Experiment configuration and YAML parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Parsed experiment configuration from a YAML file."""

    name: str
    description: str = ""
    seed: int = 42
    venue: Dict[str, Any] = field(default_factory=lambda: {"type": "equity", "params": {}})
    agents: List[Dict[str, Any]] = field(default_factory=list)
    recording: Dict[str, Any] = field(default_factory=lambda: {"type": "noop", "params": {}})
    evaluation: Dict[str, Any] = field(default_factory=lambda: {"metrics": []})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load experiment config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text())

        exp = data.get("experiment", {})
        return cls(
            name=exp.get("name", Path(path).stem),
            description=exp.get("description", ""),
            seed=exp.get("seed", 42),
            venue=data.get("venue", {"type": "equity", "params": {}}),
            agents=data.get("agents", []),
            recording=data.get("recording", {"type": "noop", "params": {}}),
            evaluation=data.get("evaluation", {"metrics": []}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dict (for recording/logging)."""
        return {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "seed": self.seed,
            },
            "venue": self.venue,
            "agents": self.agents,
            "recording": self.recording,
            "evaluation": self.evaluation,
        }
