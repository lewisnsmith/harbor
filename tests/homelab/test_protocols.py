"""Tests for harbor.homelab.agent — protocols, adapters, registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.agents.base_agent import AgentConfig
from harbor.agents.rule_agents import MomentumAgent, MeanReversionAgent, VolTargetAgent
from harbor.homelab.agent.protocols import Observable, Configurable
from harbor.homelab.agent.adapters import LegacyAgentAdapter
from harbor.homelab.agent.registry import AGENT_REGISTRY, build_agents
from harbor.homelab.venue.protocol import VenueSnapshot


def _make_snapshot(n_assets: int = 5, step: int = 0, lookback: int = 30) -> VenueSnapshot:
    """Helper to build a test VenueSnapshot."""
    rng = np.random.default_rng(42)
    return VenueSnapshot(
        timestamp=pd.Timestamp("2020-01-02"),
        step=step,
        assets=[f"asset_{i}" for i in range(n_assets)],
        prices=100.0 + rng.standard_normal(n_assets),
        returns=rng.standard_normal(n_assets) * 0.01,
        volume=np.zeros(n_assets),
        spread=np.full(n_assets, 0.02),
        returns_history=rng.standard_normal((lookback, n_assets)) * 0.01,
        market_type="equity",
    )


class TestLegacyAgentAdapter:
    def test_wraps_momentum_agent(self):
        inner = MomentumAgent(AgentConfig(name="mom"), n_assets=5)
        adapter = LegacyAgentAdapter(inner)
        assert adapter.name == "mom"
        assert isinstance(adapter, Observable)

    def test_act_returns_orders(self):
        inner = MomentumAgent(AgentConfig(name="mom"), n_assets=5, lookback=10)
        adapter = LegacyAgentAdapter(inner)
        snap = _make_snapshot(n_assets=5, lookback=30)
        orders = adapter.act(snap)
        assert orders.shape == (5,)
        assert np.isfinite(orders).all()

    def test_observe_decide_cycle(self):
        inner = VolTargetAgent(AgentConfig(name="vol"), n_assets=3)
        adapter = LegacyAgentAdapter(inner)
        snap = _make_snapshot(n_assets=3, lookback=30)
        adapter.observe(snap)
        weights = adapter.decide()
        assert weights.shape == (3,)

    def test_all_rule_agents_adaptable(self):
        for cls in [MomentumAgent, MeanReversionAgent, VolTargetAgent]:
            inner = cls(AgentConfig(name="test"), n_assets=4)
            adapter = LegacyAgentAdapter(inner)
            snap = _make_snapshot(n_assets=4)
            orders = adapter.act(snap)
            assert orders.shape == (4,)


class TestAgentRegistry:
    def test_registry_has_all_rule_types(self):
        assert "momentum" in AGENT_REGISTRY
        assert "mean_reversion" in AGENT_REGISTRY
        assert "vol_target" in AGENT_REGISTRY

    def test_build_agents_single(self):
        specs = [{"type": "momentum", "params": {"lookback": 10}}]
        agents = build_agents(specs, n_assets=5)
        assert len(agents) == 1
        assert isinstance(agents[0], Observable)

    def test_build_agents_with_count(self):
        specs = [{"type": "momentum", "count": 3}]
        agents = build_agents(specs, n_assets=5)
        assert len(agents) == 3
        names = [a.name for a in agents]
        assert names == ["momentum_0", "momentum_1", "momentum_2"]

    def test_build_agents_mixed(self):
        specs = [
            {"type": "momentum", "count": 2},
            {"type": "mean_reversion", "count": 1},
            {"type": "vol_target", "count": 1},
        ]
        agents = build_agents(specs, n_assets=5)
        assert len(agents) == 4

    def test_build_agents_unknown_type(self):
        specs = [{"type": "nonexistent"}]
        with pytest.raises(ValueError, match="Unknown agent type"):
            build_agents(specs, n_assets=5)
