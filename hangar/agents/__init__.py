"""hangar.agents — Agent Simulation Framework

Defines, configures, and runs heterogeneous agent populations in a
market simulation environment. Agents interact to generate synthetic
order flow and price impact data for causal testing of ABF hypotheses.

Agent types:
- Rule-based: momentum, mean-reversion, vol-targeting
- ML-based: DRL behavioral agents (from hangar.ml.behavior_agents)
- LLM-based: autonomous agents making allocation decisions (planned)
"""

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
