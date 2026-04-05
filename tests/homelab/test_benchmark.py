"""Integration tests: full YAML → run → verify outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hangar.homelab.config import ExperimentConfig
from hangar.homelab.runner import ExperimentRunner


BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"


class TestMomentumBaseline:
    @pytest.fixture()
    def result(self, tmp_path: Path):
        config = ExperimentConfig.from_yaml(BENCHMARKS_DIR / "momentum_baseline.yaml")
        # Override recording to use tmp_path
        config.recording = {"type": "jsonl", "params": {"output_dir": str(tmp_path)}}
        return ExperimentRunner(config).run()

    def test_shapes(self, result):
        assert result.prices.shape == (501, 10)
        assert result.returns.shape == (501, 10)
        assert result.orders.shape == (500, 10)

    def test_agent_count(self, result):
        assert len(result.agent_weights) == 10
        assert len(result.agent_returns) == 10

    def test_metrics_present(self, result):
        for metric in ["annualized_vol", "sharpe_ratio", "max_drawdown", "sortino_ratio"]:
            assert metric in result.metrics

    def test_deterministic(self, tmp_path: Path):
        config = ExperimentConfig.from_yaml(BENCHMARKS_DIR / "momentum_baseline.yaml")
        config.recording = {"type": "noop", "params": {}}
        r1 = ExperimentRunner(config).run()
        r2 = ExperimentRunner(config).run()
        np.testing.assert_array_equal(r1.prices.values, r2.prices.values)


class TestMixedPopulation:
    @pytest.fixture()
    def result(self, tmp_path: Path):
        config = ExperimentConfig.from_yaml(BENCHMARKS_DIR / "mixed_population.yaml")
        config.recording = {"type": "jsonl", "params": {"output_dir": str(tmp_path)}}
        return ExperimentRunner(config).run()

    def test_shapes(self, result):
        assert result.prices.shape == (501, 10)
        assert result.orders.shape == (500, 10)

    def test_agent_count(self, result):
        # 5 momentum + 3 mean_reversion + 2 vol_target = 10
        assert len(result.agent_weights) == 10

    def test_metrics_present(self, result):
        for metric in ["annualized_vol", "sharpe_ratio", "max_drawdown", "sortino_ratio"]:
            assert metric in result.metrics

    def test_trace_file(self, result, tmp_path: Path):
        trace = tmp_path / "mixed_population" / "trace.jsonl"
        assert trace.exists()
        lines = trace.read_text().strip().split("\n")
        # start + 500 steps + end = 502
        assert len(lines) == 502


class TestBenchmarkDivergence:
    """Verify different benchmarks produce different results."""

    def test_momentum_vs_mixed_diverge(self):
        cfg1 = ExperimentConfig.from_yaml(BENCHMARKS_DIR / "momentum_baseline.yaml")
        cfg1.recording = {"type": "noop", "params": {}}
        cfg2 = ExperimentConfig.from_yaml(BENCHMARKS_DIR / "mixed_population.yaml")
        cfg2.recording = {"type": "noop", "params": {}}

        r1 = ExperimentRunner(cfg1).run()
        r2 = ExperimentRunner(cfg2).run()

        # Same venue seed but different agent populations → different order flow → different prices
        assert not np.array_equal(r1.orders.values, r2.orders.values)
