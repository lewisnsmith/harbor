"""Tests for harbor.ml.behavior_agents.environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.ml.behavior_agents.environment import EnvConfig, PortfolioEnv


@pytest.fixture
def market_returns() -> pd.DataFrame:
    """Generate 500 days x 5 assets of synthetic returns."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=500)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    data = rng.normal(0.0005, 0.015, size=(500, 5))
    return pd.DataFrame(data, index=dates, columns=tickers)


def test_env_reset_returns_valid_observation(market_returns):
    env = PortfolioEnv(market_returns)
    obs, info = env.reset(seed=42)

    assert "returns_window" in obs
    assert "current_weights" in obs
    assert "rolling_vol" in obs
    assert "drawdown" in obs

    assert obs["returns_window"].shape == (60, 5)
    assert obs["current_weights"].shape == (5,)
    assert obs["rolling_vol"].shape == (5,)
    assert obs["drawdown"].shape == (1,)


def test_env_step_returns_valid_tuple(market_returns):
    env = PortfolioEnv(market_returns)
    env.reset(seed=42)

    action = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "portfolio_return" in info
    assert "turnover" in info


def test_env_target_action_mode(market_returns):
    config = EnvConfig(action_mode="target")
    env = PortfolioEnv(market_returns, config=config)
    env.reset(seed=42)

    # Unequal weights should get normalized
    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    obs, _, _, _, info = env.step(action)

    weights = obs["current_weights"]
    assert abs(weights.sum() - 1.0) < 1e-5


def test_env_delta_action_mode(market_returns):
    config = EnvConfig(action_mode="delta")
    env = PortfolioEnv(market_returns, config=config)
    obs, _ = env.reset(seed=42)

    initial_weights = obs["current_weights"].copy()
    # Small delta should change weights
    action = np.array([0.1, -0.1, 0.0, 0.05, -0.05], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    new_weights = obs["current_weights"]
    assert abs(new_weights.sum() - 1.0) < 1e-5
    assert not np.allclose(new_weights, initial_weights, atol=1e-3)


def test_env_episode_terminates(market_returns):
    config = EnvConfig(max_episode_steps=10)
    env = PortfolioEnv(market_returns, config=config)
    env.reset(seed=42)

    for _i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert truncated or terminated


def test_env_transaction_costs_applied(market_returns):
    """Large weight changes should incur transaction costs."""
    config = EnvConfig(transaction_cost_bps=100.0)  # 1% per unit turnover
    env = PortfolioEnv(market_returns, config=config)
    env.reset(seed=42)

    # Large rebalance: go from equal weight to concentrated
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)

    assert info["transaction_cost"] > 0
    assert info["turnover"] > 0


def test_env_properties(market_returns):
    env = PortfolioEnv(market_returns)
    assert env.n_assets == 5
    assert env.asset_names == ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
