"""harbor.homelab.batch — BatchRunner for running multiple experiments."""

from __future__ import annotations

from pathlib import Path
from typing import List

from harbor.homelab.config import ExperimentConfig
from harbor.homelab.runner import ExperimentResult, ExperimentRunner


class BatchRunner:
    """Run multiple experiment configs sequentially, collecting results."""

    def __init__(self, configs: List[ExperimentConfig]) -> None:
        self.configs = configs

    @classmethod
    def from_yaml_paths(cls, paths: List[str | Path]) -> BatchRunner:
        """Load configs from multiple YAML files."""
        return cls([ExperimentConfig.from_yaml(p) for p in paths])

    def run_all(self) -> List[ExperimentResult]:
        """Run all experiments and return results."""
        results = []
        for config in self.configs:
            runner = ExperimentRunner(config)
            results.append(runner.run())
        return results
