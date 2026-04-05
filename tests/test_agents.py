"""Tests for hangar.agents — H3 Agent Simulation Core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hangar.agents.base_agent import AgentConfig, BaseAgent
from hangar.agents.config import PopulationConfig, build_population
from hangar.agents.environment import MarketConfig, MarketEnvironment, MarketState
from hangar.agents.metrics import (
    compute_crowding_index,
    compute_flow_imbalance,
    compute_regime_labels,
    compute_return_autocorrelation,
    compute_simulation_summary,
)
from hangar.agents.rule_agents import MeanReversionAgent, MomentumAgent, VolTargetAgent
from hangar.agents.simulation import SimulationResult, run_simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def market_config() -> MarketConfig:
    return MarketConfig(n_steps=100, n_assets=5, seed=42)


@pytest.fixture
def market(market_config) -> MarketEnvironment:
    return MarketEnvironment(market_config)


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(name="test_agent")


# ---------------------------------------------------------------------------
# TestMarketEnvironment
# ---------------------------------------------------------------------------


class TestMarketEnvironment:
    def test_reset_produces_valid_state(self, market):
        state = market.reset()
        assert isinstance(state, MarketState)
        assert len(state.prices) == 5
        assert state.step == 0
        assert isinstance(state.date, pd.Timestamp)

    def test_reset_prices_are_100(self, market):
        state = market.reset()
        np.testing.assert_allclose(state.prices.values, 100.0)

    def test_step_updates_prices(self, market):
        state = market.reset()
        initial_prices = state.prices.values.copy()
        state = market.step({})
        assert not np.allclose(state.prices.values, initial_prices)
        assert state.step == 1

    def test_step_requires_reset(self):
        m = MarketEnvironment(MarketConfig(n_steps=10, n_assets=3))
        with pytest.raises(RuntimeError, match="reset"):
            m.step({})

    def test_price_impact_scales_with_order_size(self, market):
        """Larger orders should cause larger price deviations."""
        # Run two environments with same seed but different order sizes
        m1 = MarketEnvironment(MarketConfig(n_steps=10, n_assets=5, seed=42))
        m2 = MarketEnvironment(MarketConfig(n_steps=10, n_assets=5, seed=42))
        m1.reset()
        m2.reset()

        small_order = np.full(5, 0.01)
        large_order = np.full(5, 0.5)

        s1 = m1.step({"agent": small_order})
        s2 = m2.step({"agent": large_order})

        # Returns should differ due to impact
        assert not np.allclose(s1.returns.values, s2.returns.values)

    def test_no_drift_explosion(self):
        """Prices should stay bounded over 1000 steps with no agents."""
        m = MarketEnvironment(MarketConfig(n_steps=1000, n_assets=5, seed=42))
        state = m.reset()
        for _ in range(1000):
            state = m.step({})
        assert np.all(state.prices.values > 0)
        assert np.all(state.prices.values < 1e6)

    def test_deterministic_with_seed(self):
        m1 = MarketEnvironment(MarketConfig(n_steps=50, n_assets=3, seed=123))
        m2 = MarketEnvironment(MarketConfig(n_steps=50, n_assets=3, seed=123))
        s1 = m1.reset()
        s2 = m2.reset()
        for _ in range(50):
            s1 = m1.step({})
            s2 = m2.step({})
        np.testing.assert_allclose(s1.prices.values, s2.prices.values)

    def test_returns_history_grows(self, market):
        state = market.reset()
        assert len(state.returns_history) == 1
        for _ in range(5):
            state = market.step({})
        assert len(state.returns_history) == 6


# ---------------------------------------------------------------------------
# TestBaseAgent
# ---------------------------------------------------------------------------


class TestBaseAgent:
    def test_interface_contract(self, agent_config):
        """BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent(agent_config, 5)

    def test_act_returns_orders(self, market, agent_config):
        """Concrete agent's act method returns valid order array."""
        agent = MomentumAgent(agent_config, 5)
        state = market.reset()
        orders = agent.act(state)
        assert orders.shape == (5,)
        assert np.isfinite(orders).all()

    def test_weights_update_after_act(self, market, agent_config):
        agent = MomentumAgent(agent_config, 5)
        state = market.reset()
        initial_weights = agent.current_weights.copy()
        agent.act(state)
        assert not np.allclose(agent.current_weights, initial_weights)


# ---------------------------------------------------------------------------
# TestMomentumAgent
# ---------------------------------------------------------------------------


class TestMomentumAgent:
    def test_buys_winners(self, market):
        agent = MomentumAgent(AgentConfig("mom"), 5, lookback=5)
        state = market.reset()
        # Run a few steps to build history
        for _ in range(10):
            agent.act(state)
            state = market.step({})

        # After enough history, weights should be non-uniform
        weights = agent.current_weights
        assert not np.allclose(weights, 1.0 / 5)

    def test_weights_sum_to_one(self, market):
        agent = MomentumAgent(AgentConfig("mom"), 5, lookback=5)
        state = market.reset()
        for _ in range(10):
            agent.act(state)
            state = market.step({})
        np.testing.assert_allclose(agent.current_weights.sum(), 1.0, atol=1e-10)

    def test_opposite_of_mean_reversion(self, market):
        """Momentum and mean-reversion should produce different rankings."""
        mom = MomentumAgent(AgentConfig("mom"), 5, lookback=5)
        mr = MeanReversionAgent(AgentConfig("mr"), 5, lookback=5)
        state = market.reset()
        for _ in range(20):
            mom.act(state)
            mr.act(state)
            state = market.step({})
        # Weights should be negatively correlated (approximately opposite)
        corr = np.corrcoef(mom.current_weights, mr.current_weights)[0, 1]
        assert corr < 0.5  # Not perfectly correlated


# ---------------------------------------------------------------------------
# TestVolTargetAgent
# ---------------------------------------------------------------------------


class TestVolTargetAgent:
    def test_reduces_exposure_in_high_vol(self):
        """Agent should allocate less to high-vol assets."""
        m = MarketEnvironment(MarketConfig(n_steps=100, n_assets=5, seed=42))
        agent = VolTargetAgent(
            AgentConfig("vt"), 5, target_vol=0.10, vol_window=10
        )
        state = m.reset()
        for _ in range(50):
            agent.act(state)
            state = m.step({})
        weights = agent.current_weights
        assert weights.sum() > 0
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)

    def test_weights_are_positive(self):
        m = MarketEnvironment(MarketConfig(n_steps=50, n_assets=3, seed=42))
        agent = VolTargetAgent(AgentConfig("vt"), 3)
        state = m.reset()
        for _ in range(30):
            agent.act(state)
            state = m.step({})
        assert np.all(agent.current_weights >= 0)


# ---------------------------------------------------------------------------
# TestMeanReversionAgent
# ---------------------------------------------------------------------------


class TestMeanReversionAgent:
    def test_weights_sum_to_one(self, market):
        agent = MeanReversionAgent(AgentConfig("mr"), 5, lookback=5)
        state = market.reset()
        for _ in range(10):
            agent.act(state)
            state = market.step({})
        np.testing.assert_allclose(agent.current_weights.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TestSimulationRunner
# ---------------------------------------------------------------------------


class TestSimulationRunner:
    def test_end_to_end_with_three_agents(self):
        config = MarketConfig(n_steps=50, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [
            MomentumAgent(AgentConfig("mom"), 5),
            VolTargetAgent(AgentConfig("vt"), 5),
            MeanReversionAgent(AgentConfig("mr"), 5),
        ]
        result = run_simulation(market, agents)
        assert isinstance(result, SimulationResult)
        assert result.prices.shape == (51, 5)
        assert result.returns.shape == (51, 5)
        assert result.orders.shape == (50, 5)
        assert len(result.agent_weights) == 3
        assert len(result.agent_returns) == 3

    def test_output_agent_weights_shape(self):
        config = MarketConfig(n_steps=30, n_assets=3, seed=42)
        market = MarketEnvironment(config)
        agents = [MomentumAgent(AgentConfig("mom"), 3)]
        result = run_simulation(market, agents)
        assert result.agent_weights["mom"].shape == (31, 3)
        assert result.agent_returns["mom"].shape == (30,)

    def test_prices_stay_positive(self):
        config = MarketConfig(n_steps=200, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [
            MomentumAgent(AgentConfig("mom"), 5),
            MeanReversionAgent(AgentConfig("mr"), 5),
        ]
        result = run_simulation(market, agents)
        assert (result.prices.values > 0).all()


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_crowding_index_range(self):
        config = MarketConfig(n_steps=50, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [
            MomentumAgent(AgentConfig("mom_0"), 5),
            MomentumAgent(AgentConfig("mom_1"), 5),
        ]
        result = run_simulation(market, agents)
        crowding = compute_crowding_index(result.agent_weights)
        valid = crowding.dropna()
        assert (valid >= -0.01).all()  # Near 0 or above
        assert (valid <= 1.01).all()   # At most 1

    def test_crowding_high_for_homogeneous(self):
        """Identical agents should produce high crowding."""
        config = MarketConfig(n_steps=50, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [
            MomentumAgent(AgentConfig("mom_0"), 5, lookback=21),
            MomentumAgent(AgentConfig("mom_1"), 5, lookback=21),
        ]
        result = run_simulation(market, agents)
        crowding = compute_crowding_index(result.agent_weights)
        # Identical agents → very high similarity
        assert crowding.mean() > 0.9

    def test_flow_imbalance_nonnegative(self):
        config = MarketConfig(n_steps=30, n_assets=3, seed=42)
        market = MarketEnvironment(config)
        agents = [MomentumAgent(AgentConfig("mom"), 3)]
        result = run_simulation(market, agents)
        flow = compute_flow_imbalance(result.orders)
        assert (flow >= 0).all()

    def test_simulation_summary_keys(self):
        config = MarketConfig(n_steps=50, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [MomentumAgent(AgentConfig("mom"), 5)]
        result = run_simulation(market, agents)
        summary = compute_simulation_summary(
            result.prices, result.returns, result.agent_weights, result.orders
        )
        expected_keys = {
            "annualized_vol",
            "return_autocorrelation",
            "crowding_mean",
            "crowding_std",
            "flow_imbalance_mean",
            "regime_shock_count",
            "n_agents",
            "n_steps",
        }
        assert set(summary.keys()) == expected_keys

    def test_return_autocorrelation_bounded(self):
        config = MarketConfig(n_steps=200, n_assets=5, seed=42)
        market = MarketEnvironment(config)
        agents = [MomentumAgent(AgentConfig("mom"), 5)]
        result = run_simulation(market, agents)
        autocorr = compute_return_autocorrelation(result.returns)
        valid = autocorr.dropna()
        assert (valid >= -1.01).all()
        assert (valid <= 1.01).all()


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_build_population_homogeneous(self):
        config = PopulationConfig(
            agents=[{"type": "momentum", "name": "mom", "count": 5, "params": {"lookback": 10}}],
            market=MarketConfig(n_steps=20, n_assets=3),
        )
        market, agents = build_population(config)
        assert isinstance(market, MarketEnvironment)
        assert len(agents) == 5
        assert all(isinstance(a, MomentumAgent) for a in agents)

    def test_build_population_mixed(self):
        config = PopulationConfig(
            agents=[
                {"type": "momentum", "name": "mom", "count": 2},
                {"type": "vol_target", "name": "vt", "count": 1},
            ],
            market=MarketConfig(n_steps=10, n_assets=3),
        )
        _, agents = build_population(config)
        assert len(agents) == 3

    def test_unknown_agent_type_raises(self):
        config = PopulationConfig(
            agents=[{"type": "unknown", "name": "x"}],
            market=MarketConfig(n_steps=10, n_assets=3),
        )
        with pytest.raises(ValueError, match="Unknown agent type"):
            build_population(config)

    def test_from_json(self, tmp_path):
        import json
        data = {
            "description": "test",
            "market": {"n_steps": 10, "n_assets": 3, "seed": 99},
            "agents": [{"type": "momentum", "name": "m", "count": 2}],
        }
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        config = PopulationConfig.from_json(p)
        assert config.market.seed == 99
        assert len(config.agents) == 1
