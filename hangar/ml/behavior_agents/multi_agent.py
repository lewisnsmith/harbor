"""hangar.ml.behavior_agents.multi_agent — Multi-agent portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from hangar.ml.behavior_agents.agent import agent_as_weight_func


@dataclass
class AgentSpec:
    """Specification for one agent in a multi-agent simulation."""

    name: str
    checkpoint_path: str
    env_config: Dict[str, Any] = field(default_factory=dict)
    reward_shaper_config: Optional[Dict[str, Any]] = None


@dataclass
class MultiAgentResult:
    """Results from a multi-agent simulation."""

    agent_weights: Dict[str, pd.DataFrame]
    agent_returns: Dict[str, pd.Series]
    aggregate_weights: pd.DataFrame
    crowding_proxy: pd.Series
    correlation_series: pd.Series


def run_multi_agent_simulation(
    returns: pd.DataFrame,
    agent_specs: List[AgentSpec],
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    rebalance_frequency: int = 1,
    transaction_cost_bps: float = 5.0,
) -> MultiAgentResult:
    """Run multiple trained agents on the same market data simultaneously.

    Each agent independently observes the same returns and generates weights.
    No price impact is modeled.

    Parameters
    ----------
    returns
        Daily return panel shared by all agents.
    agent_specs
        List of agent specifications (checkpoint paths + configs).
    start_idx
        Starting index in the returns panel.
    end_idx
        Ending index (exclusive). Defaults to end of panel.
    rebalance_frequency
        Days between rebalance decisions.
    transaction_cost_bps
        Transaction cost in basis points.

    Returns
    -------
    MultiAgentResult
        Weight histories, returns, and crowding diagnostics for all agents.
    """
    if end_idx is None:
        end_idx = len(returns)

    assets = returns.columns.tolist()
    dates = returns.index[start_idx:end_idx]
    n_assets = len(assets)

    # Build weight functions for each agent
    weight_funcs = {}
    for spec in agent_specs:
        wf = agent_as_weight_func(
            spec.checkpoint_path, returns, env_config=spec.env_config
        )
        weight_funcs[spec.name] = wf

    # Initialize storage
    agent_weights: Dict[str, pd.DataFrame] = {}
    agent_returns: Dict[str, pd.Series] = {}

    for spec in agent_specs:
        agent_weights[spec.name] = pd.DataFrame(
            np.nan, index=dates, columns=assets, dtype=float
        )
        agent_returns[spec.name] = pd.Series(
            np.nan, index=dates, dtype=float, name=spec.name
        )

    # Run simulation
    current_weights = {
        spec.name: pd.Series(1.0 / n_assets, index=assets)
        for spec in agent_specs
    }

    for i, _date in enumerate(dates):
        abs_idx = start_idx + i
        should_rebalance = i % rebalance_frequency == 0 and abs_idx >= 60

        for spec in agent_specs:
            name = spec.name
            if should_rebalance:
                lookback_start = max(0, abs_idx - 252)
                lookback = returns.iloc[lookback_start:abs_idx]
                new_w = weight_funcs[name](lookback, current_weights[name])
                turnover = float(np.abs(new_w - current_weights[name]).sum())
                tc = turnover * transaction_cost_bps / 10_000.0
                current_weights[name] = new_w
            else:
                tc = 0.0

            daily_r = float(np.dot(
                current_weights[name].reindex(assets, fill_value=0.0).values,
                returns.iloc[abs_idx].reindex(assets, fill_value=0.0).values,
            ))
            agent_returns[name].iloc[i] = daily_r - tc
            reindexed = current_weights[name].reindex(assets, fill_value=0.0)
            agent_weights[name].iloc[i] = reindexed.values

    # Aggregate weights (equal-weighted average across agents)
    all_w = np.stack([agent_weights[s.name].values for s in agent_specs], axis=0)
    aggregate_weights = pd.DataFrame(
        np.nanmean(all_w, axis=0), index=dates, columns=assets
    )

    # Crowding proxy: average pairwise weight similarity
    crowding_proxy = compute_weight_similarity(agent_weights)

    # Correlation series: rolling 21-day pairwise return correlation
    ret_df = pd.DataFrame(agent_returns)
    if len(agent_specs) >= 2:
        rolling_corr = ret_df.rolling(21).corr()
        # Average off-diagonal correlation
        corr_values = []
        for d in dates:
            if d in rolling_corr.index.get_level_values(0):
                corr_mat = rolling_corr.loc[d]
                n = len(corr_mat)
                if n >= 2:
                    off_diag = corr_mat.values[np.triu_indices(n, k=1)]
                    corr_values.append(float(np.nanmean(off_diag)))
                else:
                    corr_values.append(float("nan"))
            else:
                corr_values.append(float("nan"))
        correlation_series = pd.Series(corr_values, index=dates, name="avg_correlation")
    else:
        correlation_series = pd.Series(float("nan"), index=dates, name="avg_correlation")

    return MultiAgentResult(
        agent_weights=agent_weights,
        agent_returns=agent_returns,
        aggregate_weights=aggregate_weights,
        crowding_proxy=crowding_proxy,
        correlation_series=correlation_series,
    )


def compute_weight_similarity(
    agent_weights: Dict[str, pd.DataFrame],
    *,
    method: str = "cosine",
) -> pd.Series:
    """Compute average pairwise weight similarity across agents over time.

    Parameters
    ----------
    agent_weights
        Dict mapping agent name to weight DataFrame (dates x tickers).
    method
        Similarity metric: ``"cosine"`` or ``"correlation"``.

    Returns
    -------
    pd.Series
        Time series of average pairwise similarity, indexed by date.
    """
    names = list(agent_weights.keys())
    if len(names) < 2:
        first = agent_weights[names[0]]
        return pd.Series(1.0, index=first.index, name="weight_similarity")

    ref = agent_weights[names[0]]
    dates = ref.index
    similarities = []

    for idx in range(len(dates)):
        pair_sims = []
        for a, b in combinations(names, 2):
            w_a = agent_weights[a].iloc[idx].values
            w_b = agent_weights[b].iloc[idx].values

            if np.any(np.isnan(w_a)) or np.any(np.isnan(w_b)):
                continue

            if method == "cosine":
                norm_a = np.linalg.norm(w_a)
                norm_b = np.linalg.norm(w_b)
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(w_a, w_b) / (norm_a * norm_b))
                else:
                    sim = 0.0
            elif method == "correlation":
                if np.std(w_a) > 0 and np.std(w_b) > 0:
                    sim = float(np.corrcoef(w_a, w_b)[0, 1])
                else:
                    sim = 0.0
            else:
                raise ValueError(f"Unknown method: {method!r}")

            pair_sims.append(sim)

        similarities.append(float(np.mean(pair_sims)) if pair_sims else float("nan"))

    return pd.Series(similarities, index=dates, name="weight_similarity")
