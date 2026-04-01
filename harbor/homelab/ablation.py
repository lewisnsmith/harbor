"""harbor.homelab.ablation — AblationRunner for parameter sweeps."""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict, List, Tuple

from harbor.homelab.config import ExperimentConfig
from harbor.homelab.runner import ExperimentResult, ExperimentRunner


class AblationRunner:
    """Generate and run experiments from a base config + parameter grid.

    Example grid:
        {"venue.params.temporary_impact": [0.05, 0.1, 0.2],
         "agents.0.count": [5, 10, 20]}

    Each combination becomes a separate experiment.
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        grid: Dict[str, List[Any]],
    ) -> None:
        self.base_config = base_config
        self.grid = grid

    def generate_configs(self) -> List[Tuple[Dict[str, Any], ExperimentConfig]]:
        """Generate all config variants from the parameter grid."""
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        configs = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            config = self._apply_params(params)
            configs.append((params, config))

        return configs

    def run_all(self) -> List[Tuple[Dict[str, Any], ExperimentResult]]:
        """Run all ablation variants and return (params, result) pairs."""
        results = []
        for params, config in self.generate_configs():
            runner = ExperimentRunner(config)
            result = runner.run()
            results.append((params, result))
        return results

    def _apply_params(self, params: Dict[str, Any]) -> ExperimentConfig:
        """Apply parameter overrides to a copy of the base config."""
        config_dict = self.base_config.to_dict()

        for dotpath, value in params.items():
            self._set_nested(config_dict, dotpath, value)

        # Reconstruct ExperimentConfig from modified dict
        exp = config_dict.get("experiment", {})
        suffix = "_".join(f"{k.split('.')[-1]}={v}" for k, v in params.items())
        return ExperimentConfig(
            name=f"{exp.get('name', 'ablation')}_{suffix}",
            description=exp.get("description", ""),
            seed=exp.get("seed", 42),
            venue=config_dict.get("venue", {}),
            agents=config_dict.get("agents", []),
            recording=config_dict.get("recording", {"type": "noop", "params": {}}),
            evaluation=config_dict.get("evaluation", {"metrics": []}),
        )

    @staticmethod
    def _set_nested(d: dict, dotpath: str, value: Any) -> None:
        """Set a value in a nested dict using dot notation (e.g., 'venue.params.n_assets')."""
        keys = dotpath.split(".")
        current = d
        for key in keys[:-1]:
            if key.isdigit():
                current = current[int(key)]
            else:
                current = current.setdefault(key, {})
        final_key = keys[-1]
        if final_key.isdigit():
            current[int(final_key)] = value
        else:
            current[final_key] = value
