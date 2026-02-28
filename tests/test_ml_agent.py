"""Tests for harbor.ml.behavior_agents.agent (lightweight, no full training)."""

from __future__ import annotations

from harbor.ml.behavior_agents.agent import AgentConfig


def test_agent_config_defaults():
    """Verify default config values are sensible."""
    config = AgentConfig()
    assert config.algorithm == "PPO"
    assert config.framework == "torch"
    assert config.learning_rate > 0
    assert config.gamma > 0 and config.gamma <= 1.0
    assert config.num_workers == 0  # local-only default


def test_agent_config_custom():
    """Verify custom config values are set correctly."""
    config = AgentConfig(
        algorithm="PPO",
        hidden_sizes=(128, 128),
        learning_rate=1e-4,
        gamma=0.95,
    )
    assert config.hidden_sizes == (128, 128)
    assert config.learning_rate == 1e-4
    assert config.gamma == 0.95
