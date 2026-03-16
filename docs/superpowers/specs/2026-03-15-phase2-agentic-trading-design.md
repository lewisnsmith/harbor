# Phase 2: Agentic Trading & AI Impact — Design Spec

**Date:** 2026-03-15
**Author:** Lewis Smith
**Status:** Draft
**Scope:** H3-H5 + A3-A4 (ML validation through agent simulation)

---

## 1. Overview

Phase 2 covers the next major execution block of HARBOR: validating the ML scaffolding, building an agent simulation framework, and running both empirical and simulation-based research on how AI agents affect market behavior.

**Approach:** Two-Track Parallel with Convergence

- **Track 1 (Empirical):** A3 — Q2 crowding/correlation analysis using real market data. Can start immediately (depends only on H2, which is complete).
- **Track 2 (Simulation):** H3 → H4 → H5 — validate ML models, validate behavioral agents, build agent simulation framework.
- **Convergence:** A4 — compare simulation output against real-data findings. The central question: do simulated agent populations reproduce the patterns found empirically?

```
Track 1 (Empirical)          Track 2 (Simulation)
---------------------        ---------------------
A3: Q2 Crowding/             H3: Validate ML models
    Correlation                  (NN vol vs GARCH,
    (real market data)           DRL vs buy-and-hold)
         |                            |
         |                   H4: Behavioral agents
         |                       validated + integrated
         |                            |
         |                   H5: Agent simulation
         |                       framework (rule-based
         |                       + ML + LLM agent types)
         |                            |
         +-------- A4 ---------------+
              Agent-General Empirical Extensions
              (simulated agents reproduce real patterns?)
```

**Timeline:** ~6-7 months of work, parallelism reducing wall-clock time.

**Critical path:** Track 2 (H3→H4→H5 = ~16 weeks) + A4 (~6 weeks) = ~5.5 months minimum. Track 1 (A3 = 8 weeks) finishes well before Track 2, so Track 2 is the bottleneck. Any H3/H4/H5 delay directly pushes A4. If H3 Gate 2 fails and requires simplification, add 2-4 weeks to the critical path.

---

## 2. Track 1 — A3 (Q2: Agent Convergence & Instability)

**Goal:** Empirically test whether behavioral convergence among trading agents amplifies drawdowns and destabilizes correlations — using real market data only.

**Module:** `harbor.abf.q2`

### 2.1 Proxy Construction

| Proxy | Definition | Source |
|-------|-----------|--------|
| Momentum crowding | Cross-sectional return dispersion (low dispersion = crowded) | Computed from price data |
| Signal similarity | Rolling correlation of factor exposures across sectors/size groups | Computed from returns + factor loadings |
| ETF flow pressure | Aggregate ETF inflow/outflow as systematic activity measure | External data (ETF.com or similar) |
| Vol-control exposure | Risk-parity deleveraging indicator | Already exists from H2 (`vol_control_pressure_proxy`) |

### 2.2 Detection

- **Correlation spike detector:** Periods where realized cross-asset correlation exceeds trailing 75th percentile.
- **Drawdown severity classifier:** Regime labels for "normal drawdown" vs "amplified drawdown."

### 2.3 Regressions

- Crowding proxies → correlation expansion (predictive regressions with 1-5 day lead)
- Crowding proxies → drawdown severity (controlling for baseline volatility, liquidity)
- Granger causality tests: does crowding lead correlation spikes?

### 2.4 Target Figures

1. Crowding proxy time series with correlation spike overlay
2. Drawdown severity conditional on crowding regime (box plot or density)
3. Lead-lag heatmap: crowding proxy → correlation expansion at various horizons
4. Robustness: pre/post-2020, high/low liquidity splits

### 2.5 Exit Criteria

- At least one crowding proxy predicts correlation expansion out-of-sample (AUC >0.65)
- Granger causality at 1-5 day lead with p<0.05
- Robustness to 2 sub-sample splits
- Reproducible in <5 commands
- Draft text for a paper section on Q2

**Note on ETF flow proxy:** The ETF flow pressure proxy requires external data. A3 can pass exit criteria without it (3 of 4 proxies are computable from existing data). ETF flows are a "nice to have" enhancement, not a blocker.

### 2.6 Sprint Breakdown

| Sprint | Duration | Focus |
|--------|----------|-------|
| A3-S1 | 2 weeks | Proxy construction + correlation spike detector |
| A3-S2 | 2 weeks | Regressions + Granger causality |
| A3-S3 | 2 weeks | Figures, robustness checks |
| A3-S4 | 2 weeks | Write-up, reproducibility, integration with risk engine |

---

## 3. Track 2 — H3 (ML Validation)

**Goal:** Promote existing experimental ML scaffolding to validated status by benchmarking against classical baselines.

**Modules:** `harbor.ml.volatility`, `harbor.ml.behavior_agents`

### 3.1 Gate 1: NN Volatility Forecasters

- Benchmark LSTM/GRU vol forecasters against GARCH(1,1) and EWMA baselines
- Metrics: RMSE, MAE, QLIKE (standard vol forecast loss), directional accuracy
- Walk-forward validation on rolling out-of-sample windows
- **Pass criterion:** NN forecaster beats at least one classical baseline on at least 2 of 4 metrics across 2+ test windows
- **If fail:** Document result honestly, keep module as "experimental," proceed with classical vol models as fallback

### 3.2 Gate 2: DRL Agent Baseline

- Benchmark actor-critic DRL agent against buy-and-hold and equal-weight baselines
- Metrics: Sharpe ratio, max drawdown, turnover
- Run on existing Gym-style portfolio environment with SP500 subset
- **Pass criterion:** DRL agent achieves risk-adjusted return within 80% of buy-and-hold (demonstrates learned behavior, not random noise)
- **If fail:** Simplify reward function, reduce action space, document learnings

### 3.3 Sprint Breakdown

| Sprint | Duration | Focus |
|--------|----------|-------|
| H3-S1 | 2 weeks | Classical baselines (GARCH, EWMA), walk-forward eval harness |
| H3-S2 | 2 weeks | NN vol benchmark runs, DRL agent benchmark, Gate 1+2 decisions |
| H3-S3 | 2 weeks | Integration: plug validated forecasters into risk engine, document results |

### 3.4 Exit Criteria

- Gate 1 and Gate 2 evaluated with documented results (pass or fail)
- Validated models integrated into `harbor.risk` vol forecasting pipeline
- Results notebook committed with reproducible commands

---

## 4. Track 2 — H4 (Behavioral Agents Validated)

**Goal:** Validate that behavioral reward shaping produces meaningfully different agent behavior, and integrate agents into the backtest pipeline.

**Depends on:** H3 Gate 2 passed (or simplified DRL agent working)

### 4.1 Behavioral Differentiation Tests

- Run each behavioral reward type (loss-averse, overconfident, return-chaser, disposition) against neutral baseline agent
- Measure: position concentration, turnover, drawdown profile, holding period distribution
- **Pass criterion:** At least 3 of 4 behavioral types produce statistically distinguishable portfolio characteristics from baseline (KS test on return distributions, p<0.05)

### 4.2 Multi-Agent Configuration

- Configure populations of 2-5 agents with mixed behavioral types
- Verify agents can run concurrently in the same environment
- Output: per-agent position history and aggregate market impact metrics

### 4.3 Backtest Integration

- Agents used as "strategy generators" in the backtest engine
- Agent-generated portfolios evaluated with standard HARBOR metrics

### 4.4 Sprint Breakdown

| Sprint | Duration | Focus |
|--------|----------|-------|
| H4-S1 | 2 weeks | Behavioral differentiation tests, per-agent metrics |
| H4-S2 | 2 weeks | Multi-agent config, backtest integration, exit criteria |

### 4.5 Exit Criteria

- 3+ behavioral types produce distinguishable behavior
- Multi-agent config runs without errors
- Agent portfolios evaluable through standard backtest pipeline

---

## 5. Track 2 — H5 (Agent Simulation Framework)

**Goal:** Build a market simulation environment where heterogeneous agent populations interact, generating synthetic order flow and price impact data analyzable with the same ABF pipelines used on real data.

**Module:** `harbor.agents`

**Depends on:** H4 (validated behavioral agents as one agent type)

### 5.1 Architecture

```
harbor/agents/
    __init__.py
    environment.py      # Market simulation environment
    base_agent.py       # Abstract agent interface
    rule_agents.py      # Vol-targeting, momentum, mean-reversion
    ml_agents.py        # Wrapper around harbor.ml.behavior_agents
    llm_agents.py       # LLM-based agent (stub for future)
    population.py       # Agent population manager + config loader
    metrics.py          # Aggregate metrics: crowding, flow, price impact
    configs/            # YAML/JSON population configurations
```

### 5.2 Core Components

**Market Environment (`environment.py`):**
- Simplified price-impact model (not full order book)
- Accepts agent orders each timestep, computes clearing price with impact
- Price impact: linear temporary impact + square-root permanent impact (standard microstructure)
- Tracks: price history, volume, order flow imbalance, realized volatility
- Config-driven: timesteps, initial price, impact parameters, noise level

**Agent Interface (`base_agent.py`):**
```python
class BaseAgent(ABC):
    def observe(self, market_state: MarketState) -> None: ...
    def act(self) -> Order: ...
    def update(self, fill: Fill) -> None: ...
```

**Agent Types:**

| Type | Source | Behavior |
|------|--------|----------|
| Rule-based | `rule_agents.py` | Momentum, vol-targeting, mean-reversion with configurable parameters |
| ML-based | `ml_agents.py` | Wraps validated DRL agents from H4 |
| LLM-based | `llm_agents.py` | Stub — interface defined, `NotImplementedError` raised. Phase 3 scope. |

**Population Manager (`population.py`):**
- Loads agent population configs (agent counts, type mix, parameter distributions)
- Runs simulation loop: initialize → observe → act → clear → update
- Outputs: DataFrame of prices, volumes, per-agent positions, aggregate metrics

**Simulation Metrics (`metrics.py`):**
- Crowding index: position similarity across agents (cosine similarity of holdings vectors)
- Flow imbalance: net buy/sell pressure per timestep
- Synthetic vol-control proxy: aggregate exposure changes from vol-targeting agents
- Correlation structure: rolling cross-agent return correlation
- Designed to be directly comparable to A3's real-data proxies

### 5.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Price formation | Impact model (not full order book) | Order book is overkill; impact model captures dynamics we care about; standard in agent-based finance lit |
| Agent heterogeneity | Configurable parameter distributions | Same agent type with different parameters creates realistic diversity |
| LLM agents | Stub only in Phase 2 | Fascinating but scope-risky; interface defined now, implementation in Phase 3 |
| Output format | Same DataFrame structure as real market data | Enables reuse of entire ABF analysis pipeline without modification |

### 5.4 Experiment Configurations (Planned)

1. **Homogeneous momentum:** 50 momentum agents → does crowding emerge?
2. **Mixed population:** 20 momentum + 20 vol-target + 10 mean-reversion → correlation dynamics
3. **Stress injection:** External vol shock applied to mixed population → cascading deleveraging?
4. **Convergence test:** Start with diverse parameters, let agents learn → do they converge?

### 5.5 Sprint Breakdown

| Sprint | Duration | Focus |
|--------|----------|-------|
| H5-S1 | 2 weeks | Market environment + base agent interface + rule-based agents |
| H5-S2 | 2 weeks | Population manager, ML agent wrapper, simulation loop |
| H5-S3 | 2 weeks | Metrics module, experiment configs, end-to-end simulation runs |

### 5.6 Exit Criteria

- Configurable multi-agent simulation runs end-to-end
- Rule-based + ML-based agents produce synthetic price/volume data
- Metrics module computes crowding, flow imbalance, synthetic vol-control proxy
- Output format compatible with ABF analysis pipelines
- At least 2 experiment configurations produce non-trivial results

---

## 6. Convergence — A4 (Agent-General Empirical Extensions)

**Goal:** Compare synthetic agent simulation data (H5) against real market patterns (A3) to test whether agent populations reproduce the crowding, momentum, and instability patterns found empirically.

**Module:** `harbor.abf.q2` (extended) + `harbor.abf.agent_validation`

**Depends on:** A3 and H5 both complete

### 6.1 The Core Question

"Do simulated agent populations, with no access to real market data, independently produce the same statistical signatures we found in actual markets?"

- If yes → strong causal evidence that agent convergence drives these patterns.
- If no → patterns may be driven by fundamentals, not agent behavior. Also publishable.

### 6.2 Pattern Matching Framework

- Compute exact A3 metrics (crowding proxy, correlation spike detection, drawdown severity) on H5's synthetic data
- Distribution comparison: KS test, Wasserstein distance between synthetic and real metric distributions
- Sensitivity analysis: which simulation parameters most affect pattern reproduction?

### 6.3 Causal Experiments

These are experiments you *cannot* run on real markets — the simulation framework's unique contribution.

- **Ablation:** Remove vol-targeting agents from simulation → do correlation spikes disappear?
- **Dose-response:** Increase momentum agent proportion from 10% to 80% → does crowding monotonically increase?
- **Heterogeneity test:** High-diversity vs low-diversity agent populations → does convergence produce the worst outcomes?

### 6.4 Target Figures

1. Side-by-side: real vs synthetic crowding proxy distributions
2. Ablation results: which agent types drive which patterns
3. Dose-response curves: agent concentration → crowding severity
4. Phase diagram: agent diversity x market stress → outcome severity

### 6.5 Sprint Breakdown

| Sprint | Duration | Focus |
|--------|----------|-------|
| A4-S1 | 2 weeks | Pattern matching framework, metric computation on synthetic data |
| A4-S2 | 2 weeks | Causal experiments (ablation, dose-response) |
| A4-S3 | 2 weeks | Figures, statistical comparison, write-up |

### 6.6 Exit Criteria

- At least one synthetic metric distribution statistically consistent with real data (KS p>0.05)
- Ablation experiments show at least one agent type whose removal significantly changes a pattern
- Results documented in paper-quality figures with reproducible commands
- Draft text for "simulation validates empirical findings" paper section

---

## 7. Testing Strategy (Phase 2 Overall)

### New Test Files

| File | Coverage | Est. Tests |
|------|----------|-----------|
| `tests/test_abf_q2.py` | A3 proxy construction, correlation detection, regressions | ~25-30 |
| `tests/test_ml_validation.py` | H3 benchmark harness, gate evaluation logic | ~15-20 |
| `tests/test_agents.py` | H5 environment, agent interface, population manager, metrics | ~30-40 |
| `tests/test_agent_validation.py` | A4 pattern matching, distribution comparison | ~10-15 |

**Estimated total:** 80-105 new tests (bringing project total to ~270-290)

### Test Principles

- All simulation tests use fixed random seeds for reproducibility
- Agent tests verify interface contracts, not specific learned behavior
- Regression tests compare against committed baseline results
- No live API calls — all external data mocked

---

## 8. File Change Summary

### New Files

| File | Purpose |
|------|---------|
| `harbor/abf/q2/proxies.py` | Crowding proxy construction |
| `harbor/abf/q2/detection.py` | Correlation spike + drawdown severity detection |
| `harbor/abf/q2/regressions.py` | Predictive regressions + Granger causality |
| `harbor/abf/q2/figures.py` | Q2 figure generation |
| `harbor/abf/agent_validation/` | A4 pattern matching and causal experiments |
| `harbor/agents/environment.py` | Market simulation environment |
| `harbor/agents/base_agent.py` | Abstract agent interface |
| `harbor/agents/rule_agents.py` | Rule-based agents |
| `harbor/agents/ml_agents.py` | ML agent wrapper |
| `harbor/agents/llm_agents.py` | LLM agent stub |
| `harbor/agents/population.py` | Population manager |
| `harbor/agents/metrics.py` | Simulation metrics |
| `configs/agents/` | Agent population configuration files |
| `experiments/abf_q2_main.py` | Q2 experiment orchestrator |
| `experiments/agent_sim_demo.py` | Agent simulation demo script |

### Modified Files

| File | Change |
|------|--------|
| `docs/plan.md` | Add H3-H5, A3-A4 detail; update vision, architecture, milestones |
| `docs/abf-prd.md` | Broaden language, add agent simulation section, retail integration |
| `CHANGELOG.md` | Add v0.3.0-dev entries |
| `harbor/agents/__init__.py` | Replace stub with real exports |
| `harbor/abf/q2/__init__.py` | Replace stub with real exports |

### Unchanged

- All existing code modules, tests, configs, experiments
- H1/H2 implementation (187 tests stay passing)
- ABF Q1 pipeline and results

---

## 9. Updated Milestone Table

| Timeframe | HARBOR Focus | ABF Focus | Status |
|-----------|-------------|-----------|--------|
| 0-3 months | H1: Data, HRP, Monte Carlo, backtest | A1: Spec + data, Q1 prep | Complete |
| 3-6 months | H2: Advanced risk, scenarios | A2: Q1 full execution + draft figures | Complete |
| 6-8 months | H3: ML validation (NN vol, DRL) | A3: Q2 proxies + crowding regressions | **Next** |
| 8-10 months | H4: Behavioral agent validation | A3: Q2 figures, robustness, write-up | Planned |
| 10-13 months | H5: Agent simulation framework | A4: Pattern matching (sim vs real) | Planned |
| 13-15 months | H5: Agent experiments | A4: Causal experiments + write-up | Planned |
| 15-18 months | H6: Retail impact analysis | A5: Retail impact quantification | Planned |
| 18+ months | LLM agents, polish | Writing, external validation, publication | Planned |
