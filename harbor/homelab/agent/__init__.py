"""harbor.homelab.agent — Protocol-based agent interface and adapters."""

from harbor.homelab.agent.protocols import (
    BudgetAware,
    Configurable,
    Observable,
    ToolUser,
)
from harbor.homelab.agent.adapters import LegacyAgentAdapter
from harbor.homelab.agent.registry import AGENT_REGISTRY, build_agents

__all__ = [
    "Observable",
    "Configurable",
    "ToolUser",
    "BudgetAware",
    "LegacyAgentAdapter",
    "AGENT_REGISTRY",
    "build_agents",
]
