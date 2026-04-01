"""harbor.homelab.agent.protocols — Composable agent protocols.

Agents implement only the protocols they need:
- Observable: core interface (observe + decide + act)
- Configurable: parameters can be set from experiment config
- ToolUser: agent uses external tools (risk models, data feeds)
- BudgetAware: agent has compute/API budget constraints
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np

from harbor.homelab.venue.protocol import VenueSnapshot


@runtime_checkable
class Observable(Protocol):
    """Core agent protocol: receives venue observations and produces orders."""

    @property
    def name(self) -> str: ...

    def observe(self, snapshot: VenueSnapshot) -> None: ...

    def decide(self) -> np.ndarray: ...

    def act(self, snapshot: VenueSnapshot) -> np.ndarray: ...


@runtime_checkable
class Configurable(Protocol):
    """Agent whose parameters can be get/set from experiment config."""

    def get_params(self) -> Dict[str, Any]: ...

    def set_params(self, params: Dict[str, Any]) -> None: ...


@runtime_checkable
class ToolUser(Protocol):
    """Agent that uses external tools (risk models, data, etc.)."""

    def available_tools(self) -> List[str]: ...

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class BudgetAware(Protocol):
    """Agent with compute/API budget constraints."""

    @property
    def budget_remaining(self) -> float: ...

    def deduct_budget(self, amount: float) -> None: ...
