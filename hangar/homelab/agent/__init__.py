"""hangar.homelab.agent — Protocol-based agent interface and adapters."""

from hangar.homelab.agent.protocols import (
    BudgetAware,
    Configurable,
    Observable,
    ToolUser,
)
from hangar.homelab.agent.adapters import LegacyAgentAdapter
from hangar.homelab.agent.registry import AGENT_REGISTRY, build_agents

__all__ = [
    "Observable",
    "Configurable",
    "ToolUser",
    "BudgetAware",
    "LegacyAgentAdapter",
    "AGENT_REGISTRY",
    "build_agents",
]
