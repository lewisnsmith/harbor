"""Tests for AutonomousAgent and RandomAgent."""

from __future__ import annotations

import numpy as np
import pytest

from hangar.agents.base_agent import AgentConfig
from hangar.agents.autonomous_agent import AutonomousAgent, RandomAgent
from hangar.agents.environment import MarketConfig, MarketEnvironment
from hangar.homelab.config import ExperimentConfig
from hangar.homelab.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_agent(agent, n_steps=20, seed=42):
    """Run an agent through a market for n_steps, return final state."""
    market = MarketEnvironment(MarketConfig(n_steps=n_steps, n_assets=5, seed=seed))
    state = market.reset()
    for _ in range(n_steps):
        orders = agent.act(state)
        state = market.step({agent.name: orders})
    return agent, state


# ---------------------------------------------------------------------------
# TestAutonomousAgent
# ---------------------------------------------------------------------------


class TestAutonomousAgent:
    def test_weights_sum_to_one(self):
        agent = AutonomousAgent(AgentConfig("auto"), 5, risk_appetite=0.5, reactivity=0.5)
        agent, _ = _run_agent(agent, n_steps=20)
        np.testing.assert_allclose(agent.current_weights.sum(), 1.0, atol=1e-8)

    def test_weights_all_nonnegative(self):
        agent = AutonomousAgent(AgentConfig("auto"), 5, risk_appetite=0.8, reactivity=0.8)
        agent, _ = _run_agent(agent, n_steps=30)
        assert np.all(agent.current_weights >= -1e-10)

    def test_lookback_from_reactivity(self):
        # reactivity=1.0 → lookback=5
        a1 = AutonomousAgent(AgentConfig("a1"), 5, reactivity=1.0)
        assert a1.lookback == 5

        # reactivity=0.5 → lookback=round(5+55*0.5)=round(32.5)=32
        a2 = AutonomousAgent(AgentConfig("a2"), 5, reactivity=0.5)
        assert a2.lookback == 32

        # reactivity=0.0 → lookback=round(5+55*1.0)=60
        a3 = AutonomousAgent(AgentConfig("a3"), 5, reactivity=0.0)
        assert a3.lookback == 60

    def test_target_vol_from_risk_appetite(self):
        # risk=0.0 → target_vol=0.01
        a1 = AutonomousAgent(AgentConfig("a1"), 5, risk_appetite=0.0)
        assert abs(a1.target_vol - 0.01) < 1e-10

        # risk=1.0 → target_vol=0.10
        a2 = AutonomousAgent(AgentConfig("a2"), 5, risk_appetite=1.0)
        assert abs(a2.target_vol - 0.10) < 1e-10

    def test_equal_weights_at_start(self):
        """Insufficient history → equal weights."""
        agent = AutonomousAgent(AgentConfig("auto"), 5, risk_appetite=0.5, reactivity=0.5)
        market = MarketEnvironment(MarketConfig(n_steps=10, n_assets=5, seed=42))
        state = market.reset()
        # First act with only 1 row of history → equal weights
        agent.act(state)
        # After first step with damping from zero, weights should be moving toward 1/n
        assert agent.current_weights.sum() > 0

    def test_rebalance_damping(self):
        """Low reactivity agent should move weights more slowly than high reactivity."""
        market_cfg = MarketConfig(n_steps=30, n_assets=5, seed=42)

        slow = AutonomousAgent(AgentConfig("slow"), 5, risk_appetite=0.5, reactivity=0.1)
        fast = AutonomousAgent(AgentConfig("fast"), 5, risk_appetite=0.5, reactivity=0.9)

        m1 = MarketEnvironment(market_cfg)
        m2 = MarketEnvironment(market_cfg)
        s1 = m1.reset()
        s2 = m2.reset()

        slow_deltas = []
        fast_deltas = []
        for _ in range(20):
            prev_slow = slow.current_weights.copy()
            prev_fast = fast.current_weights.copy()
            o1 = slow.act(s1)
            o2 = fast.act(s2)
            slow_deltas.append(np.abs(o1).sum())
            fast_deltas.append(np.abs(o2).sum())
            s1 = m1.step({slow.name: o1})
            s2 = m2.step({fast.name: o2})

        # Fast agent should have larger cumulative weight changes
        assert sum(fast_deltas) > sum(slow_deltas)

    def test_deterministic(self):
        """Same market + same personality = same weights."""
        a1 = AutonomousAgent(AgentConfig("a1"), 5, risk_appetite=0.7, reactivity=0.3)
        a2 = AutonomousAgent(AgentConfig("a2"), 5, risk_appetite=0.7, reactivity=0.3)
        _run_agent(a1, n_steps=20, seed=42)
        _run_agent(a2, n_steps=20, seed=42)
        np.testing.assert_allclose(a1.current_weights, a2.current_weights, atol=1e-10)

    def test_different_personalities_diverge(self):
        """Different params → different weights."""
        a1 = AutonomousAgent(AgentConfig("a1"), 5, risk_appetite=0.1, reactivity=0.1)
        a2 = AutonomousAgent(AgentConfig("a2"), 5, risk_appetite=0.9, reactivity=0.9)
        _run_agent(a1, n_steps=30, seed=42)
        _run_agent(a2, n_steps=30, seed=42)
        assert not np.allclose(a1.current_weights, a2.current_weights, atol=1e-3)

    def test_signal_blend_extremes(self):
        """reactivity=1.0 should be momentum-dominant, =0.0 should be mean-rev-dominant."""
        mom_agent = AutonomousAgent(AgentConfig("mom"), 5, risk_appetite=0.5, reactivity=1.0)
        mr_agent = AutonomousAgent(AgentConfig("mr"), 5, risk_appetite=0.5, reactivity=0.0)
        _run_agent(mom_agent, n_steps=30, seed=42)
        _run_agent(mr_agent, n_steps=30, seed=42)
        # They should produce meaningfully different weight allocations
        assert not np.allclose(mom_agent.current_weights, mr_agent.current_weights, atol=0.01)

    def test_vol_scaling_effect(self):
        """Higher risk_appetite → larger weight dispersion (more concentrated bets)."""
        low = AutonomousAgent(AgentConfig("low"), 5, risk_appetite=0.1, reactivity=0.5)
        high = AutonomousAgent(AgentConfig("high"), 5, risk_appetite=1.0, reactivity=0.5)
        _run_agent(low, n_steps=30, seed=42)
        _run_agent(high, n_steps=30, seed=42)
        # Higher risk appetite should produce more dispersed weights
        low_disp = low.current_weights.std()
        high_disp = high.current_weights.std()
        assert high_disp > low_disp


# ---------------------------------------------------------------------------
# TestRandomAgent
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def test_weights_sum_to_one(self):
        agent = RandomAgent(AgentConfig("rand"), 5, seed=42)
        agent, _ = _run_agent(agent, n_steps=10)
        np.testing.assert_allclose(agent.current_weights.sum(), 1.0, atol=1e-8)

    def test_weights_nonnegative(self):
        agent = RandomAgent(AgentConfig("rand"), 5, seed=42)
        agent, _ = _run_agent(agent, n_steps=10)
        assert np.all(agent.current_weights >= -1e-10)

    def test_different_each_step(self):
        """Random agent should produce different weights each step."""
        agent = RandomAgent(AgentConfig("rand"), 5, seed=42)
        market = MarketEnvironment(MarketConfig(n_steps=10, n_assets=5, seed=42))
        state = market.reset()

        weights_history = []
        for _ in range(5):
            agent.act(state)
            weights_history.append(agent.current_weights.copy())
            state = market.step({})

        # At least some steps should have different weights
        all_same = all(np.allclose(weights_history[0], w) for w in weights_history[1:])
        assert not all_same

    def test_deterministic_with_seed(self):
        a1 = RandomAgent(AgentConfig("r1"), 5, seed=123)
        a2 = RandomAgent(AgentConfig("r2"), 5, seed=123)
        _run_agent(a1, n_steps=10, seed=42)
        _run_agent(a2, n_steps=10, seed=42)
        np.testing.assert_allclose(a1.current_weights, a2.current_weights, atol=1e-10)


# ---------------------------------------------------------------------------
# TestRegistryIntegration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_autonomous_in_registry(self):
        from hangar.homelab.agent.registry import build_agents
        agents = build_agents(
            [{"type": "autonomous", "params": {"risk_appetite": 0.5, "reactivity": 0.5}}],
            n_assets=5,
        )
        assert len(agents) == 1
        assert agents[0].name == "autonomous"

    def test_random_in_registry(self):
        from hangar.homelab.agent.registry import build_agents
        agents = build_agents(
            [{"type": "random", "params": {"seed": 42}}],
            n_assets=5,
        )
        assert len(agents) == 1
        assert agents[0].name == "random"

    def test_end_to_end_runner(self):
        """ExperimentRunner with autonomous agents produces valid result."""
        config = ExperimentConfig(
            name="test_autonomous",
            seed=42,
            venue={"type": "equity", "params": {"n_steps": 30, "n_assets": 5}},
            agents=[
                {"type": "autonomous", "name": "auto_1",
                 "params": {"risk_appetite": 0.3, "reactivity": 0.7}},
                {"type": "autonomous", "name": "auto_2",
                 "params": {"risk_appetite": 0.8, "reactivity": 0.2}},
                {"type": "random", "name": "rand",
                 "params": {"seed": 0}},
            ],
            evaluation={"metrics": ["crowding_index"]},
        )
        result = ExperimentRunner(config).run()
        assert result.prices.shape == (31, 5)
        assert len(result.agent_weights) == 3
        assert "auto_1" in result.agent_weights
        assert "rand" in result.agent_weights
        # Weights should be valid
        for name, w_df in result.agent_weights.items():
            final_w = w_df.iloc[-1].values
            assert np.all(np.isfinite(final_w))
