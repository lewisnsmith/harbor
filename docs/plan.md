# HARBOR & ABF — Architecture and Status

Project Owner: Lewis Smith
Last Updated: 2026-04-01

---

## 0. Vision

HARBOR is a math-first platform for studying whether autonomous agents create statistically exploitable structure in markets. The core thesis: autonomous agents (LLM-powered, RL-based, tool-using) are qualitatively different from traditional systematic algorithms — they reason, adapt, and interact strategically — and when deployed at scale, they create emergent coordination, manufactured regimes, and adversarial dynamics that traditional finance cannot explain.

**HARBOR's triple role:**

1. **Infrastructure** — agents use HARBOR's risk models, portfolio construction, and data pipeline
2. **Participant** — HARBOR-as-agent competes against autonomous agents, testing whether regime-awareness provides edge
3. **Origin story** — the asset management algorithm that sparked the research question

ABF (Artificial Behavioral Finance) is the research track that uses HARBOR's infrastructure and the agent simulation framework to test whether autonomous trading agents create novel market dynamics.

---

## 1. Architecture: Five Layers

HARBOR is organized as five layers. Each layer wraps or builds on the one below it.

### Layer 1 — Market / Venue

**Module:** `harbor/homelab/venue/`

Normalizes different market types into a unified interface. Currently implemented:

- `VenueSnapshot` — standardized state schema: timestamp, assets, prices, returns, volume, spread, returns_history, market_type, metadata
- `EquityVenue` — adapter wrapping `harbor.agents.MarketEnvironment` as a venue, synthesizing volume and spread from order flow and volatility
- `Venue` protocol — `reset(seed) → VenueSnapshot`, `step(orders) → VenueSnapshot`

All venues emit `VenueSnapshot`. The runner consumes only the protocol — venues are pluggable.

### Layer 2 — Agent

**Modules:** `harbor/homelab/agent/`, `harbor/agents/`

Agents are pluggable policies. The homelab agent layer is protocol-based: agents compose only the interfaces they need.

**Protocols (`harbor/homelab/agent/protocols.py`):**
- `Observable` — core interface: `observe()`, `decide()`, `act()` → orders
- `Configurable` — `get_params()` / `set_params()` for ablation sweeps
- `ToolUser` — `available_tools()` / `use_tool()` for agents that call external services
- `BudgetAware` — `budget_remaining` / `deduct_budget()` for API-constrained agents

**Adapters:** `LegacyAgentAdapter` bridges existing `BaseAgent` subclasses (Momentum, MeanReversion, VolTarget) to the `Observable` protocol without rewriting them.

**Registry:** `build_agents(config, n_assets)` constructs agents from YAML config entries.

**Rule agents (`harbor/agents/rule_agents.py`):**
- `MomentumAgent` — trend-following with lookback window
- `MeanReversionAgent` — mean-reversion with lookback window
- `VolTargetAgent` — volatility-targeting weight scaler

### Layer 3 — Portfolio / Risk

**Modules:** `harbor/risk/`, `harbor/portfolio/`

The existing math-heavy core. Unchanged by the restructure. Callable by agents as tool-using services.

- **Risk:** sample and Ledoit-Wolf shrinkage covariance, HRP, Monte Carlo VaR/CVaR, regime detection (HMM), scenario stress tests, factor/cluster decomposition
- **Portfolio:** mean-variance, risk parity, HRP allocation with configurable constraints

### Layer 4 — Experiment / Evaluation

**Module:** `harbor/homelab/`

The reproducible experiment infrastructure — the "homelab heart."

**Config (`config.py`):**
- `ExperimentConfig` — dataclass with: name, seed, venue block, agents list, recording block, evaluation block
- `ExperimentConfig.from_yaml(path)` — loads from YAML; `to_dict()` serializes for recording

**Runner (`runner.py`):**
- `ExperimentRunner` — orchestrates: seed derivation → venue → agents → recorder → simulation loop → metrics → finalize
- `ExperimentResult` — prices, returns, agent_weights, agent_returns, orders, metrics, elapsed_seconds
- Seed derivation via `np.random.SeedSequence` — deterministic child seeds for venue and agents

**Batch / Ablation:**
- `BatchRunner` — runs multiple `ExperimentConfig`s sequentially, collects results
- `AblationRunner` — generates all combinations from a base config + parameter grid (dot-path notation: `"venue.params.n_assets": [5, 10, 20]`)

**Recording:**
- `Recorder` protocol — `start_experiment()`, `record_step()`, `end_experiment()`
- `NoopRecorder` — silent (default)
- `JsonlRecorder` — writes one JSONL record per step to `output_dir/<experiment_id>.jsonl`; designed for Flight integration

**Evaluation:**
- `MetricsRegistry` — named metric functions, `compute_all(names, prices, returns, agent_weights, orders, agent_returns)`
- `ExperimentSummary` — aggregates result sets into summary DataFrames

**Results store (`results/store.py`):**
- `ResultsStore` — persists `ExperimentResult` objects with metadata

**CLI (`__main__.py`):**
```bash
python -m harbor.homelab experiment.yaml
```

### Layer 5 — Exploitation

Deferred. Intended for: cross-market arbitrage, prediction-market sum-of-probabilities checks, crowding/fading strategies, inventory-aware market making. The key thesis is that the exploiter is often simpler than the agents it exploits.

---

## 2. Completed Work

### H1 — Core Quant Stack ✅

- `harbor.data` — S&P 500 universe loaders (survivorship-bias-aware fallback), price loader, risk-free rate proxy, local Parquet/pickle cache
- `harbor.risk` — sample/shrinkage covariance, HRP, Monte Carlo VaR/CVaR
- `harbor.portfolio` — mean-variance, risk parity, HRP allocation interfaces
- `harbor.backtest` — cross-sectional engine with transaction costs and core metrics
- End-to-end script: `experiments/h1_end_to_end_hrp_backtest.py`

### H2 — Advanced Risk & Simulation ✅

- Regime-aware covariance estimators
- Student-t and factor-driven Monte Carlo engines
- Config-driven stress scenario runner (vol spikes, correlation spikes, sector crashes)
- Risk decomposition by factor and cluster
- Pluggable risk engine interface
- Demo: `experiments/h2_risk_engine_demo.py`

### Agent Simulation Core (`harbor/agents/`) ✅

- `MarketEnvironment` — price-impact model (linear temporary + square-root permanent), order matching, state management
- `BaseAgent` — abstract interface: `observe() → decide() → act()`
- `MomentumAgent`, `MeanReversionAgent`, `VolTargetAgent` — rule-based baselines
- `MarketConfig` — population parameter configuration
- `PopulationMetrics` — crowding index, flow imbalance, regime labels
- `Simulation` — multi-agent simulation runner (pre-homelab, now wrapped by EquityVenue)

### Homelab / Experiment Infrastructure (`harbor/homelab/`) ✅

Full Layer 4 implementation. See Section 1 above.

### Benchmark Suite (`benchmarks/`) ✅

Reproducible benchmark configs for regression testing and baseline comparisons:

- `momentum_baseline.yaml` — 10 homogeneous momentum agents, 500 steps, 10 assets
- `mixed_population.yaml` — 5 momentum + 3 mean-reversion + 2 vol-target, 500 steps, 10 assets

Integration tests (`tests/homelab/test_benchmark.py`) verify shapes, metrics, determinism, trace output, and cross-benchmark divergence.

### ABF A1 + A2 ✅ (Deprecated)

- PRD written, SP500 universe and data pipeline built
- Q1 pipeline: local projections, Newey-West HAC errors, CAR computation, robustness sweep
- Results: ~1.3bps effects, economically insignificant, time-dependent (post-2020 only)
- Deprecated — code preserved in `harbor/abf/q1/` as historical baseline. This investigation revealed where to look: autonomous agents, not simple algorithms.

### ML Extensions (Experimental)

- LSTM/GRU volatility forecasters + GARCH/EWMA classical baselines
- Deep RL behavioral agents with behavioral reward shaping
- Labeled experimental — not validated against classical benchmarks

---

## 3. Current State

**Test suite:** 287 tests passing (234 core + 53 homelab)

**Runnable today:**
```bash
# Homelab experiment
python -m harbor.homelab configs/your_experiment.yaml

# H1 end-to-end baseline
python3 experiments/h1_end_to_end_hrp_backtest.py --start 2020-01-01 --max-assets 50

# H3 agent simulation demo
make h3
```

**Module inventory:**

| Module | Purpose | Status |
|--------|---------|--------|
| `harbor.data` | Data loaders and caching | Complete |
| `harbor.risk` | Risk models and scenario analysis | Complete |
| `harbor.portfolio` | Portfolio construction | Complete |
| `harbor.backtest` | Backtesting engine | Complete |
| `harbor.agents` | Agent simulation environment | Complete |
| `harbor.homelab` | Experiment infrastructure (Layer 4) | Complete |
| `harbor.ml` | Vol forecasters, RL agents | Experimental |
| `harbor.abf` | ABF research utilities | Legacy (Q1 deprecated) |

---

## 4. Cross-Cutting Practices

- **Reproducibility:** Config-driven experiments, deterministic seeds via `SeedSequence`, JSONL traces
- **Testing:** Pytest, 278 tests, CI via GitHub Actions
- **Documentation:** `README.md` (overview/quickstart), `docs/abf-prd.md` (research spec), `docs/plan.md` (this file), `docs/PROJECT_EVOLUTION.md` (project story)
- **Git hygiene:** Feature branches per module/experiment, descriptive commit messages
