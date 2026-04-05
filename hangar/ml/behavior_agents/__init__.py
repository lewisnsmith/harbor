"""
hangar.ml.behavior_agents — Deep RL behavioral portfolio agents.

Status: Experimental scaffolding (Layer 2 — Agent). Not yet integrated into
the homelab pipeline. Unit tests pass but agents have not been validated
against classical baselines or integrated into the production backtest pipeline.

Provides a Gymnasium portfolio environment, PPO-based agent training via
RLlib, behavioral reward shaping (loss aversion, overconfidence, return
chasing, disposition effect), and multi-agent simulation for ABF experiments.
"""

from hangar.ml.behavior_agents.agent import (
    AgentConfig,
    agent_as_weight_func,
    build_rllib_config,
    train_agent,
)
from hangar.ml.behavior_agents.environment import (
    EnvConfig,
    PortfolioEnv,
)
from hangar.ml.behavior_agents.multi_agent import (
    AgentSpec,
    MultiAgentResult,
    compute_weight_similarity,
    run_multi_agent_simulation,
)
from hangar.ml.behavior_agents.rewards import (
    CompositeRewardShaper,
    DispositionEffectShaper,
    LossAversionShaper,
    OverconfidenceShaper,
    ReturnChasingShaper,
    default_behavioral_shaper,
)

__all__ = [
    "AgentConfig",
    "AgentSpec",
    "CompositeRewardShaper",
    "DispositionEffectShaper",
    "EnvConfig",
    "LossAversionShaper",
    "MultiAgentResult",
    "OverconfidenceShaper",
    "PortfolioEnv",
    "ReturnChasingShaper",
    "agent_as_weight_func",
    "build_rllib_config",
    "compute_weight_similarity",
    "default_behavioral_shaper",
    "run_multi_agent_simulation",
    "train_agent",
]
