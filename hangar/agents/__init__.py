"""harbor.agents — Agent Simulation Framework

Defines, configures, and runs heterogeneous agent populations in a
market simulation environment. Agents interact to generate synthetic
order flow and price impact data for causal testing of ABF hypotheses.

Agent types:
- Rule-based: momentum, mean-reversion, vol-targeting
- ML-based: DRL behavioral agents (from harbor.ml.behavior_agents)
- LLM-based: autonomous agents making allocation decisions (planned)
"""

from harbor.agents.base_agent import AgentConfig, BaseAgent
from harbor.agents.config import PopulationConfig, build_population
from harbor.agents.environment import MarketConfig, MarketEnvironment, MarketState
from harbor.agents.metrics import (
    compute_crowding_index,
    compute_flow_imbalance,
    compute_regime_labels,
    compute_return_autocorrelation,
    compute_simulation_summary,
)
from harbor.agents.rule_agents import MeanReversionAgent, MomentumAgent, VolTargetAgent
from harbor.agents.simulation import SimulationResult, run_simulation

__all__ = [
    "AgentConfig",
    "BaseAgent",
    "MarketConfig",
    "MarketEnvironment",
    "MarketState",
    "MeanReversionAgent",
    "MomentumAgent",
    "PopulationConfig",
    "SimulationResult",
    "VolTargetAgent",
    "build_population",
    "compute_crowding_index",
    "compute_flow_imbalance",
    "compute_regime_labels",
    "compute_return_autocorrelation",
    "compute_simulation_summary",
    "run_simulation",
]
