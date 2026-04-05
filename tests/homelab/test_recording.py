"""Tests for hangar.homelab.recording — Recorder protocol, NoopRecorder, JsonlRecorder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hangar.homelab.recording.protocol import Recorder
from hangar.homelab.recording.noop import NoopRecorder
from hangar.homelab.recording.jsonl import JsonlRecorder
from hangar.homelab.venue.protocol import VenueSnapshot


def _make_snapshot(step: int = 0) -> VenueSnapshot:
    return VenueSnapshot(
        timestamp=pd.Timestamp("2020-01-02"),
        step=step,
        assets=["A", "B"],
        prices=np.array([100.0, 200.0]),
        returns=np.array([0.01, -0.02]),
        volume=np.array([1.0, 2.0]),
        spread=np.array([0.02, 0.02]),
        returns_history=np.zeros((5, 2)),
        market_type="equity",
    )


class TestNoopRecorder:
    def test_implements_protocol(self):
        r = NoopRecorder()
        assert isinstance(r, Recorder)

    def test_all_methods_noop(self):
        r = NoopRecorder()
        r.start_experiment({}, {})
        r.record_step(0, _make_snapshot(), {}, {})
        r.record_event("test", {"key": "val"})
        r.end_experiment({"done": True})
        r.flush()


class TestJsonlRecorder:
    def test_implements_protocol(self):
        r = JsonlRecorder(output_dir="/tmp", experiment_id="test")
        assert isinstance(r, Recorder)

    def test_writes_trace_file(self, tmp_path: Path):
        r = JsonlRecorder(output_dir=str(tmp_path), experiment_id="test_exp")
        r.start_experiment({"seed": 42}, {"run": "test"})
        snap = _make_snapshot(step=0)
        r.record_step(0, snap, {"a1": np.array([0.1, -0.1])}, {"vol": 0.15})
        r.record_step(1, _make_snapshot(step=1), {}, {})
        r.end_experiment({"total_steps": 2})

        trace_file = tmp_path / "test_exp" / "trace.jsonl"
        assert trace_file.exists()

        lines = trace_file.read_text().strip().split("\n")
        assert len(lines) == 4  # start + 2 steps + end

        first = json.loads(lines[0])
        assert first["type"] == "experiment_start"
        assert first["config"]["seed"] == 42

        step_line = json.loads(lines[1])
        assert step_line["type"] == "step"
        assert step_line["step"] == 0

    def test_writes_summary_json(self, tmp_path: Path):
        r = JsonlRecorder(output_dir=str(tmp_path), experiment_id="sum_test")
        r.start_experiment({}, {})
        r.end_experiment({"sharpe": 1.5, "vol": 0.12})

        summary_file = tmp_path / "sum_test" / "summary.json"
        assert summary_file.exists()
        data = json.loads(summary_file.read_text())
        assert data["sharpe"] == 1.5

    def test_record_event(self, tmp_path: Path):
        r = JsonlRecorder(output_dir=str(tmp_path), experiment_id="evt_test")
        r.start_experiment({}, {})
        r.record_event("agent_error", {"agent": "a1", "msg": "timeout"})
        r.end_experiment({})

        lines = (tmp_path / "evt_test" / "trace.jsonl").read_text().strip().split("\n")
        evt = json.loads(lines[1])
        assert evt["type"] == "event"
        assert evt["event_type"] == "agent_error"

    def test_numpy_serialization(self, tmp_path: Path):
        r = JsonlRecorder(output_dir=str(tmp_path), experiment_id="np_test")
        r.start_experiment({}, {})
        snap = _make_snapshot()
        r.record_step(0, snap, {"a": np.array([0.5, -0.5])}, {"val": np.float64(3.14)})
        r.end_experiment({})

        lines = (tmp_path / "np_test" / "trace.jsonl").read_text().strip().split("\n")
        step = json.loads(lines[1])
        assert step["prices"] == [100.0, 200.0]
