"""hangar.homelab.runner — ExperimentRunner: YAML config → run → traces + metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from hangar.homelab.agent.protocols import Observable
from hangar.homelab.agent.registry import build_agents
from hangar.homelab.config import ExperimentConfig
from hangar.homelab.evaluation.registry import MetricsRegistry
from hangar.homelab.recording.jsonl import JsonlRecorder
from hangar.homelab.recording.noop import NoopRecorder
from hangar.homelab.recording.protocol import Recorder
from hangar.homelab.venue.equity import EquityVenue
from hangar.homelab.venue.protocol import Venue, VenueSnapshot


@dataclass
class ExperimentResult:
    """Output of a single experiment run."""

    config: ExperimentConfig
    prices: pd.DataFrame
    returns: pd.DataFrame
    agent_weights: Dict[str, pd.DataFrame]
    agent_returns: Dict[str, pd.Series]
    orders: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class ExperimentRunner:
    """Runs a single experiment from config to completion.

    Orchestration:
    1. Derive child seeds from master seed via SeedSequence
    2. Instantiate Venue, agents, and Recorder
    3. Run simulation loop
    4. Compute metrics
    5. Return ExperimentResult
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        t0 = time.time()

        # 1. Seed derivation
        ss = np.random.SeedSequence(self.config.seed)
        venue_seed, agent_seed = ss.spawn(2)

        # 2. Build venue
        venue = self._build_venue(venue_seed.generate_state(1)[0])

        # 3. Build agents
        venue_params = self.config.venue.get("params", {})
        n_assets = venue_params.get("n_assets", 10)
        agents = build_agents(self.config.agents, n_assets)

        # 4. Build recorder
        recorder = self._build_recorder()

        # 5. Run simulation
        n_steps = venue_params.get("n_steps", 500)
        result = self._run_loop(venue, agents, recorder, n_steps, venue_seed.generate_state(1)[0])

        # 6. Compute metrics
        metric_names = self.config.evaluation.get("metrics", [])
        if metric_names:
            registry = MetricsRegistry()
            result.metrics = registry.compute_all(
                names=metric_names,
                prices=result.prices,
                returns=result.returns,
                agent_weights=result.agent_weights,
                orders=result.orders,
                agent_returns=result.agent_returns,
            )

        # 7. Finalize recording
        recorder.end_experiment({
            "config": self.config.to_dict(),
            "metrics": result.metrics,
            "elapsed_seconds": time.time() - t0,
        })

        result.elapsed_seconds = time.time() - t0
        return result

    def _build_venue(self, seed: int) -> Venue:
        venue_type = self.config.venue.get("type", "equity")
        venue_params = self.config.venue.get("params", {})
        if venue_type == "equity":
            venue = EquityVenue(venue_params)
            venue.reset(seed)
            return venue
        raise ValueError(f"Unknown venue type: {venue_type!r}")

    def _build_recorder(self) -> Recorder:
        rec_config = self.config.recording
        rec_type = rec_config.get("type", "noop")
        rec_params = rec_config.get("params", {})

        if rec_type == "noop":
            return NoopRecorder()
        elif rec_type == "jsonl":
            return JsonlRecorder(
                output_dir=rec_params.get("output_dir", "results"),
                experiment_id=self.config.name,
            )
        raise ValueError(f"Unknown recorder type: {rec_type!r}")

    def _run_loop(
        self,
        venue: Venue,
        agents: List[Observable],
        recorder: Recorder,
        n_steps: int,
        seed: int,
    ) -> ExperimentResult:
        """Core simulation loop."""
        # Reset venue and get initial snapshot
        snapshot = venue.reset(seed)
        n_assets = snapshot.n_assets
        asset_names = snapshot.assets

        # Pre-allocate storage
        price_records = np.zeros((n_steps + 1, n_assets))
        return_records = np.zeros((n_steps + 1, n_assets))
        order_records = np.zeros((n_steps, n_assets))
        weight_records = {a.name: np.zeros((n_steps + 1, n_assets)) for a in agents}
        ret_records = {a.name: np.zeros(n_steps) for a in agents}

        # Record initial state
        price_records[0] = snapshot.prices
        return_records[0] = 0.0

        recorder.start_experiment(self.config.to_dict(), {"n_agents": len(agents)})

        # Simulation loop
        for t in range(n_steps):
            agent_orders: Dict[str, np.ndarray] = {}
            for a in agents:
                orders = a.act(snapshot)
                agent_orders[a.name] = orders

            # Step venue
            snapshot = venue.step(agent_orders)

            # Record state
            price_records[t + 1] = snapshot.prices
            return_records[t + 1] = snapshot.returns
            order_records[t] = np.sum(list(agent_orders.values()), axis=0)

            # Per-agent tracking
            for a in agents:
                weight_records[a.name][t + 1] = a.current_weights
                port_ret = float(np.dot(a.current_weights, snapshot.returns))
                ret_records[a.name][t] = port_ret

            recorder.record_step(t, snapshot, agent_orders, {})

        # Build DataFrames
        dates = pd.bdate_range("2020-01-01", periods=n_steps + 1)
        prices = pd.DataFrame(price_records, index=dates, columns=asset_names)
        returns = pd.DataFrame(return_records, index=dates, columns=asset_names)
        orders = pd.DataFrame(order_records, index=dates[1:], columns=asset_names)

        agent_weights = {
            name: pd.DataFrame(w, index=dates, columns=asset_names)
            for name, w in weight_records.items()
        }
        agent_returns = {
            name: pd.Series(r, index=dates[1:], name=name)
            for name, r in ret_records.items()
        }

        return ExperimentResult(
            config=self.config,
            prices=prices,
            returns=returns,
            agent_weights=agent_weights,
            agent_returns=agent_returns,
            orders=orders,
        )
