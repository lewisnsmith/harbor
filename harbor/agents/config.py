"""harbor.agents.config — Population configuration and factory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from harbor.agents.base_agent import AgentConfig, BaseAgent
from harbor.agents.environment import MarketConfig, MarketEnvironment
from harbor.agents.rule_agents import MeanReversionAgent, MomentumAgent, VolTargetAgent

AGENT_REGISTRY = {
    "momentum": MomentumAgent,
    "mean_reversion": MeanReversionAgent,
    "vol_target": VolTargetAgent,
}


@dataclass
class PopulationConfig:
    """Declarative configuration for an agent population + market."""

    agents: List[Dict[str, Any]] = field(default_factory=list)
    market: MarketConfig = field(default_factory=MarketConfig)
    description: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> PopulationConfig:
        """Load from a JSON file."""
        data = json.loads(Path(path).read_text())
        market_kwargs = data.get("market", {})
        market = MarketConfig(**market_kwargs)
        return cls(
            agents=data.get("agents", []),
            market=market,
            description=data.get("description", ""),
        )


def build_population(
    config: PopulationConfig,
) -> Tuple[MarketEnvironment, List[BaseAgent]]:
    """Instantiate market environment and agents from a PopulationConfig."""
    market = MarketEnvironment(config.market)
    n_assets = config.market.n_assets

    agents: List[BaseAgent] = []
    for spec in config.agents:
        agent_type = spec["type"]
        name = spec["name"]
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
            agent = cls(agent_config, n_assets, **params)
            agents.append(agent)

    return market, agents
