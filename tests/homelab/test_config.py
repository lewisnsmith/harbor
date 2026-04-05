"""Tests for hangar.homelab.config — ExperimentConfig and YAML parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from hangar.homelab.config import ExperimentConfig


SAMPLE_YAML = """\
experiment:
  name: test_experiment
  description: A test experiment
  seed: 123

venue:
  type: equity
  params:
    n_steps: 50
    n_assets: 3
    base_volatility: 0.01

agents:
  - type: momentum
    count: 5
    params:
      lookback: 10
  - type: mean_reversion
    count: 3

recording:
  type: noop
  params: {}

evaluation:
  metrics:
    - annualized_vol
    - sharpe_ratio
"""


class TestExperimentConfig:
    def test_from_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(SAMPLE_YAML)

        config = ExperimentConfig.from_yaml(cfg_file)
        assert config.name == "test_experiment"
        assert config.seed == 123
        assert config.venue["type"] == "equity"
        assert config.venue["params"]["n_steps"] == 50
        assert len(config.agents) == 2
        assert config.agents[0]["type"] == "momentum"
        assert config.agents[0]["count"] == 5
        assert config.evaluation["metrics"] == ["annualized_vol", "sharpe_ratio"]

    def test_to_dict_roundtrip(self, tmp_path: Path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(SAMPLE_YAML)

        config = ExperimentConfig.from_yaml(cfg_file)
        d = config.to_dict()
        assert d["experiment"]["name"] == "test_experiment"
        assert d["experiment"]["seed"] == 123
        assert d["venue"]["type"] == "equity"

    def test_defaults(self):
        config = ExperimentConfig(name="minimal")
        assert config.seed == 42
        assert config.venue["type"] == "equity"
        assert config.agents == []
        assert config.recording["type"] == "noop"

    def test_minimal_yaml(self, tmp_path: Path):
        minimal = "experiment:\n  name: bare_minimum\n"
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text(minimal)

        config = ExperimentConfig.from_yaml(cfg_file)
        assert config.name == "bare_minimum"
        assert config.seed == 42
