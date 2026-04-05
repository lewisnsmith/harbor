"""Tests for hangar.homelab.runner — ExperimentRunner end-to-end."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hangar.homelab.config import ExperimentConfig
from hangar.homelab.runner import ExperimentRunner, ExperimentResult


def _make_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        name="test_run",
        seed=42,
        venue={"type": "equity", "params": {"n_steps": 50, "n_assets": 3}},
        agents=[
            {"type": "momentum", "count": 2, "params": {"lookback": 10}},
            {"type": "mean_reversion", "count": 1},
        ],
        recording={"type": "noop", "params": {}},
        evaluation={"metrics": ["annualized_vol", "sharpe_ratio"]},
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


class TestExperimentRunner:
    def test_basic_run(self):
        config = _make_config()
        runner = ExperimentRunner(config)
        result = runner.run()

        assert isinstance(result, ExperimentResult)
        assert result.prices.shape == (51, 3)  # n_steps + 1
        assert result.returns.shape == (51, 3)
        assert result.orders.shape == (50, 3)
        assert len(result.agent_weights) == 3  # 2 momentum + 1 mean_reversion
        assert len(result.agent_returns) == 3
        assert result.elapsed_seconds > 0

    def test_metrics_computed(self):
        config = _make_config()
        runner = ExperimentRunner(config)
        result = runner.run()

        assert "annualized_vol" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert isinstance(result.metrics["annualized_vol"], float)

    def test_deterministic(self):
        config = _make_config()
        r1 = ExperimentRunner(config).run()
        r2 = ExperimentRunner(config).run()

        np.testing.assert_array_equal(r1.prices.values, r2.prices.values)
        np.testing.assert_array_equal(r1.returns.values, r2.returns.values)

    def test_different_seeds_diverge(self):
        r1 = ExperimentRunner(_make_config(seed=1)).run()
        r2 = ExperimentRunner(_make_config(seed=2)).run()

        assert not np.array_equal(r1.prices.values, r2.prices.values)

    def test_no_agents(self):
        config = _make_config(agents=[], evaluation={"metrics": []})
        runner = ExperimentRunner(config)
        result = runner.run()

        assert result.prices.shape == (51, 3)
        assert len(result.agent_weights) == 0

    def test_jsonl_recording(self, tmp_path: Path):
        config = _make_config(
            recording={
                "type": "jsonl",
                "params": {"output_dir": str(tmp_path)},
            }
        )
        runner = ExperimentRunner(config)
        result = runner.run()

        trace_file = tmp_path / "test_run" / "trace.jsonl"
        assert trace_file.exists()
        lines = trace_file.read_text().strip().split("\n")
        # start + n_steps + end = 52
        assert len(lines) == 52

    def test_from_yaml(self, tmp_path: Path):
        yaml_content = """\
experiment:
  name: yaml_test
  seed: 99

venue:
  type: equity
  params:
    n_steps: 20
    n_assets: 2

agents:
  - type: vol_target
    count: 2

evaluation:
  metrics:
    - annualized_vol
"""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        config = ExperimentConfig.from_yaml(cfg_file)
        result = ExperimentRunner(config).run()

        assert result.prices.shape == (21, 2)
        assert "annualized_vol" in result.metrics
