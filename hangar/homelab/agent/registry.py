"""hangar.homelab.agent.registry — Agent type registry and factory."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from hangar.agents.autonomous_agent import AutonomousAgent, RandomAgent
from hangar.agents.base_agent import AgentConfig, BaseAgent
from hangar.agents.rule_agents import MeanReversionAgent, MomentumAgent, VolTargetAgent
from hangar.homelab.agent.adapters import LegacyAgentAdapter
from hangar.homelab.agent.protocols import Observable

# Registry of known agent types → BaseAgent subclasses
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "momentum": MomentumAgent,
    "mean_reversion": MeanReversionAgent,
    "vol_target": VolTargetAgent,
    "autonomous": AutonomousAgent,
    "random": RandomAgent,
}


def build_agents(
    agent_specs: List[Dict[str, Any]],
    n_assets: int,
) -> List[Observable]:
    """Build agent instances from experiment config specs.

    Each spec dict has:
        type: str — key in AGENT_REGISTRY
        count: int — number of instances (default 1)
        params: dict — passed to agent constructor
    """
    agents: List[Observable] = []

    for spec in agent_specs:
        agent_type = spec["type"]
        name = spec.get("name", agent_type)
        params = spec.get("params", {})
        count = spec.get("count", 1)

        cls = AGENT_REGISTRY.get(agent_type)
        if cls is None:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Available: {list(AGENT_REGISTRY.keys())}"
            )

        for i in range(count):
            agent_name = f"{name}_{i}" if count > 1 else name
            agent_config = AgentConfig(name=agent_name)
            inner = cls(agent_config, n_assets, **params)
            agents.append(LegacyAgentAdapter(inner))

    return agents
