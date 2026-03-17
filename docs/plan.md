# HARBOR & ABF — Plan and Roadmap

Project Owner: Lewis Smith
Last Updated: 2026-03-16

---

## 0. Vision

HARBOR is an asset management algorithm that evolved into an empirical research platform for studying how autonomous trading agents reshape financial markets. The core thesis: autonomous agents (LLM-powered, RL-based, tool-using) are qualitatively different from traditional systematic algorithms — they reason, adapt, and interact strategically — and when deployed at scale, they create emergent coordination, manufactured regimes, and adversarial dynamics that traditional finance cannot explain.

**HARBOR's triple role:**

1. **Infrastructure** — agents use HARBOR's risk models, portfolio construction, and data pipeline
2. **Participant** — HARBOR-as-agent competes against autonomous agents, testing whether regime-awareness provides edge
3. **Origin story** — the asset management algorithm that sparked the research question

ABF (Artificial Behavioral Finance) is the research track that uses HARBOR's infrastructure and the agent simulation framework to test whether autonomous trading agents create novel market dynamics.

Outputs:
- A modular Python library (HARBOR) with quant, ML, and agent simulation internals.
- A reproducible ABF research pipeline + working-paper-style draft on autonomous agent market dynamics.

---

## 1. Architecture Overview

High-level HARBOR modules:

- `harbor.data` – SP500 data, universe, features, Massive API client.
- `harbor.risk` – HRP, covariance estimation, Monte Carlo VaR/CVaR, regime detection, scenarios, decomposition.
- `harbor.portfolio` – optimization, constraints, allocation logic.
- `harbor.backtest` – engine, metrics, experiment runners.
- `harbor.ml.volatility` – NN volatility forecasters (experimental) + classical baselines (GARCH, EWMA).
- `harbor.ml.behavior_agents` – deep RL behavioral portfolio agents (experimental).
- `harbor.agents` – Agent simulation framework: market environment, autonomous agents, population experiments.
- `harbor.abf` – Research experiments and analysis utilities.
- `harbor.retail` – Retail impact metrics and analysis (planned).

The system has two arms: the asset management stack (data → risk → portfolio → backtest) and the research stack (agents + abf + retail). They share infrastructure and feed results into each other.

---

## 2. HARBOR Roadmap (Framework)

### 2.1 Phase H1 — Core Quant Stack ✅ COMPLETE

**Goal:** Solid, math-heavy portfolio/risk core with clean APIs.

Deliverables:
- `harbor.data`
  - SP500 survivorship-bias-free loaders (CRSP/WRDS if available; fallback acceptable but documented).
- `harbor.risk`
  - Covariance estimators (sample, shrinkage).
  - Hierarchical Risk Parity (HRP) implementation.
  - Basic Monte Carlo return generator + portfolio VaR/CVaR.
- `harbor.portfolio`
  - Mean–variance / risk-parity / HRP allocation interfaces.
- `harbor.backtest`
  - Cross-sectional backtest loop with transaction costs and standard performance/risk metrics.

Exit criteria:
- One end-to-end script: "load SP500 → build HRP portfolio → backtest → report risk metrics".

Phase H1 checklist (execution status):
- [x] `harbor.data`: point-in-time membership loader with documented fallback, price loader, risk-free loader, local cache scaffolding.
- [x] `harbor.risk`: sample/shrinkage covariance estimators, HRP implementation, Monte Carlo VaR/CVaR utilities.
- [x] `harbor.portfolio`: mean-variance, risk parity, and HRP allocation interfaces.
- [x] `harbor.backtest`: cross-sectional backtest engine with transaction costs and core metrics.
- [x] Exit-criteria script added: `experiments/h1_end_to_end_hrp_backtest.py`.
- [x] Phase H1 unit-test suite added for core data/risk/portfolio/backtest pathways.

---

### 2.2 Phase H2 — Advanced Risk & Simulation ✅ COMPLETE

**Goal:** Deepen risk modeling and scenario analysis.

Deliverables:
- `harbor.risk`
  - Regime-aware covariance (regime switches and shrinkage based on vol state).
  - Robust Monte Carlo engines (Student-t marginals, factor-driven simulation).
- Stress testing:
  - Shock scenarios (vol spikes, correlation spikes, sector crashes).
  - Risk decomposition by factor/cluster.

Exit criteria:
- Config-driven scenario runs with clear outputs (risk decomposition, stress-test reports).
- Pluggable risk engine used by ABF experiments.

Phase H2 checklist (execution status):
- [x] Regime-aware covariance estimators.
- [x] Student-t and factor-driven Monte Carlo engines.
- [x] Config-driven stress scenario runner.
- [x] Risk decomposition by factor and cluster.
- [x] Pluggable risk engine interface.
- [x] 48 H2 tests passing.
- [x] Demo script: `experiments/h2_risk_engine_demo.py`.

---

### 2.3 Phase H3 — Agent Simulation Core

**Goal:** Build the market simulation environment and agent interface — the primary research instrument.

**Status: Next.** This is the foundation for all agent-first research.

**Depends on:** H2 (complete).

**Module:** `harbor/agents/`

```
harbor/agents/
    environment.py      # Market sim: price-impact model, order matching, state
    base_agent.py       # Abstract interface: observe() → decide() → act()
    rule_agents.py      # Momentum, vol-targeting, mean-reversion (simple baselines)
    config.py           # Population configs, parameter distributions
    metrics.py          # Market-level: crowding index, flow imbalance, regime labels
```

Key design decisions:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Price formation | Impact model (linear temporary + square-root permanent) | Captures dynamics we care about; standard in agent-based finance lit |
| Agent interface | observe() → decide() → act() | Clean separation; works for all agent types |
| Rule-based agents | First implementation | Validates environment before adding complexity |
| Output format | Same DataFrame structure as real data | Enables reuse of entire ABF analysis pipeline |

Deliverables:
- `harbor.agents.environment`: Market environment with price-impact model
- `harbor.agents.base_agent`: Abstract agent interface
- `harbor.agents.rule_agents`: Momentum, vol-targeting, mean-reversion baselines
- `harbor.agents.config`: Population configuration system
- `harbor.agents.metrics`: Crowding index, flow imbalance, regime labels

Exit criteria:
- Market environment runs end-to-end with rule-based agents
- Price-impact model produces realistic price dynamics (no drift explosion, reasonable vol)
- Metrics module computes crowding, flow imbalance, regime labels
- Output format compatible with ABF analysis pipelines
- Unit tests for environment, agents, metrics

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H3-S1 | 2 weeks | Market environment + price-impact model + state management |
| H3-S2 | 2 weeks | Base agent interface + rule-based agents (momentum, vol-target, mean-rev) |
| H3-S3 | 2 weeks | Metrics module + config system + end-to-end simulation runs |

---

### 2.4 Phase H4 — Autonomous Agent Types

**Goal:** Implement the autonomous agent types that make this project distinctive — LLM agents, RL agents, and HARBOR-as-agent.

**Depends on:** H3 (simulation core running).

```
harbor/agents/
    llm_agents.py       # LLM-based agents (Claude/GPT via API, prompted with market data)
    rl_agents.py        # RL agents (wraps harbor.ml.behavior_agents, trains in environment)
    harbor_agent.py     # HARBOR-as-agent: uses harbor.risk + harbor.portfolio to trade
    adaptation.py       # Agent learning/adaptation between rounds
```

Agent types:
| Type | Source | Distinguishing Feature |
|------|--------|----------------------|
| LLM-based | `llm_agents.py` | Reasons about market state via API call, adapts via prompting |
| RL-based | `rl_agents.py` | Learns optimal policy through environment interaction (experimental baseline — not validated against classical benchmarks) |
| HARBOR | `harbor_agent.py` | Uses HARBOR's own risk/portfolio logic — the system tests itself |

Key design decisions:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM agents | Real API calls (Claude/GPT) | Mocking loses the point — real reasoning is what makes agents different |
| RL agents | Wrap existing harbor.ml.behavior_agents | Reuses validated code |
| HARBOR-as-agent | Wraps harbor.risk + harbor.portfolio | Tests the system against itself |
| Adaptation | Between-round strategy updates | Agents learn from past performance |

Exit criteria:
- LLM agents receive market state, output trading decisions via API
- RL agents train within the simulation environment
- HARBOR agent uses existing portfolio logic to trade
- Adaptation mechanism allows strategy updates between rounds
- Unit tests for each agent type (mocked API for CI)

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H4-S1 | 2 weeks | LLM agent implementation + prompt engineering + API integration |
| H4-S2 | 2 weeks | RL agent wrapper + HARBOR-as-agent + adaptation mechanism |
| H4-S3 | 2 weeks | Integration testing: all agent types in shared simulation |

---

### 2.5 Phase H5 — Population Dynamics & Experiments

**Goal:** Multi-agent population management and predefined experiment configurations for causal testing.

**Depends on:** H4 (all agent types working).

```
harbor/agents/
    population.py       # Population manager: spawn, configure, mix agent types
    experiments.py      # Predefined experiment configs
    analysis.py         # Bridge: convert sim output → ABF-compatible DataFrames
```

Experiment configurations:
1. **Homogeneous LLM:** 50 LLM agents, same base prompt → emergent coordination test
2. **Mixed autonomous:** 20 LLM + 20 RL + 10 HARBOR → interaction dynamics
3. **Stress injection:** External vol shock + mixed population → cascading behavior
4. **Adaptation test:** Agents learn between rounds → convergence or divergence
5. **HARBOR resilience:** HARBOR agent in autonomous-agent-dominated market

Exit criteria:
- Population manager configures and runs multi-agent simulations end-to-end
- At least 3 experiment configurations produce non-trivial results
- Analysis module converts simulation output to ABF-compatible DataFrames
- Results reproducible via config files and `make` targets

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H5-S1 | 2 weeks | Population manager + experiment config system |
| H5-S2 | 2 weeks | Run all 5 experiment configs + analysis bridge |
| H5-S3 | 2 weeks | Results analysis, reproducibility, documentation |

---

## 3. ABF Roadmap (Research)

ABF = research layer that uses HARBOR infrastructure and the agent simulation framework.

### 3.1 Phase A1 — Spec & Data ✅ COMPLETE

**Goal:** Lock questions, universe, and primary metrics.

Deliverables:
- PRD (`docs/abf-prd.md`) v1 done.
- SP500 universe with historical membership + clean daily panel.
- Config files for shock definitions and regime definitions.

---

### 3.2 Phase A2 — Old Q1: Shock → Persistence → Reversal ✅ COMPLETE (Deprecated)

**Goal:** Test whether vol-targeting algorithms create momentum and reversal patterns.

**Status:** Complete but deprecated. Results were weak and time-dependent (post-2020 only, ~1.3bps, economically insignificant). This investigation motivated the pivot to autonomous-agent-specific research. Code remains in `harbor/abf/q1/` as historical work. See `docs/abf-q1-research-summary.md`.

---

### 3.3 Phase A3 — Q1: Emergent Coordination

**Goal:** Test whether independently-built autonomous agents converge on similar strategies without explicit coordination.

**Depends on:** H4 (autonomous agent types working).

**Module:** `harbor/abf/q1_coordination/`

Method:
- Run H4 agent populations with diverse initial configurations (different LLM prompts, different RL training seeds, different HARBOR configs)
- Measure convergence metrics:

| Metric | Definition |
|--------|-----------|
| Position correlation | Cross-agent correlation of portfolio weights over time |
| Strategy similarity | Rolling cosine similarity of agent trade signals |
| Herding intensity | Fraction of agents on the same side of each trade |
| Convergence speed | Time (in rounds) for similarity metrics to plateau |

- Key test: do agents that start different become more similar over time?
- Empirical complement: post-2023 real-market signatures (did cross-asset correlations change character after LLM agent adoption?)

Target figures:
1. Convergence trajectories: similarity metrics over simulation rounds for each experiment config
2. Herding intensity heatmap: agent-by-agent position correlation matrix at start vs end
3. Empirical comparison: real-market correlation structure pre-2020 vs post-2023
4. Robustness: sensitivity to LLM prompt variation, RL seed variation

Exit criteria:
- Statistical evidence that independent agents converge (or clear null result)
- At least one convergence metric shows clear temporal trend in simulation
- Empirical comparison with real-market patterns
- Draft text for paper section on emergent coordination

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| A3-S1 | 2 weeks | Convergence metric computation + simulation runs |
| A3-S2 | 2 weeks | Empirical complement (real-market analysis) |
| A3-S3 | 2 weeks | Figures, robustness, write-up |

---

### 3.4 Phase A4 — Q2: Regime Manufacturing

**Goal:** Test whether autonomous agent populations CREATE market regimes that wouldn't exist without them.

**Depends on:** H5 (population experiments) and A3 (convergence findings inform regime analysis).

**Module:** `harbor/abf/q2_regimes/`

Method:
- Ablation experiments: run simulation with and without agent populations, compare regime structure using H2's regime detection tools
- Dose-response: vary agent concentration from 10% to 80% of market volume, measure regime intensity
- Regime detection: apply existing `harbor.risk` regime tools to synthetic data
- Empirical complement: compare regime frequency and intensity pre-2020 vs post-2023

Target figures:
1. Ablation comparison: regime structure with vs without agents
2. Dose-response curves: agent concentration → regime intensity
3. Regime type identification: which regimes are agent-manufactured?
4. Empirical comparison: regime frequency pre-2020 vs post-2023

Exit criteria:
- Ablation shows statistically significant regime differences with/without agents
- Dose-response shows monotonic relationship between agent concentration and regime intensity
- At least one manufactured regime type clearly identified
- Draft text for paper section

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| A4-S1 | 2 weeks | Ablation experiments + regime detection on synthetic data |
| A4-S2 | 2 weeks | Dose-response curves + empirical complement |
| A4-S3 | 2 weeks | Figures, statistical tests, write-up |

---

### 3.5 Phase A5 — Q3: Adversarial Adaptation

**Goal:** Test whether autonomous agents learn to exploit each other and whether this destabilizes prices.

**Depends on:** A4 (regime findings inform adaptation analysis).

**Module:** `harbor/abf/q3_adversarial/`

Method:
- Multi-round simulation where agents adapt strategies between rounds
- Measure: price stability over time, strategy divergence/convergence, Sharpe ratio decay
- Test whether adaptation leads to arms race (increasing instability), equilibrium (stability), or cycles
- Retail impact: how does a naive buy-and-hold investor fare as agents adapt?
- HARBOR impact: does regime-awareness mitigate the damage?

Target figures:
1. Price stability over adaptation rounds
2. Strategy divergence/convergence trajectories
3. Sharpe decay curves for different agent types
4. Retail portfolio impact during adversarial periods
5. HARBOR-as-agent performance vs naive strategies

Exit criteria:
- Clear characterization of adaptation dynamics (arms race, equilibrium, or cycles)
- Quantified retail portfolio impact during adversarial periods
- HARBOR-as-agent performance comparison
- Draft text for paper section

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| A5-S1 | 2 weeks | Multi-round adaptation experiments |
| A5-S2 | 2 weeks | Retail + HARBOR impact analysis |
| A5-S3 | 2 weeks | Figures, characterization, write-up |

---

### 3.6 Phase A6 — Retail Impact & Write-up

**Goal:** Complete paper draft with HARBOR-as-participant results and retail impact quantification.

**Depends on:** A5.

Deliverables:
- HARBOR-as-agent performance across all experiment configs
- Quantified regime-awareness benefit for retail portfolios
- Paper-quality draft: Q1 (coordination) + Q2 (regimes) + Q3 (adaptation) + retail impact
- Reproducible pipeline via `make` targets
- Public or semi-public repo with docs and notebooks aligned to the paper

Exit criteria:
- Complete paper draft with figures, tables, and statistical tests
- All experiments reproducible in <5 commands
- At least one expert review (faculty or practitioner)
- Concise writeup suitable for applications

---

## 4. Cross-Cutting Practices

- **Versioning & Changelog**
  - Semantic Versioning: `MAJOR.MINOR.PATCH`.
  - `CHANGELOG.md` updated for each tagged release.

- **Documentation**
  - `README.md` for overview and quickstart.
  - `docs/abf-prd.md` for research spec.
  - `docs/plan.md` (this file) for roadmap and phases.
  - `docs/PROJECT_EVOLUTION.md` for the full project story.

- **Git Hygiene**
  - Feature branches per module/experiment.
  - Descriptive, technical commit messages and PRs.

- **Reproducibility**
  - Config-driven experiments.
  - Minimal "one-command" scripts to reproduce key results.

---

## 5. Milestone Table (High-Level)

| Timeframe | HARBOR Focus | ABF Focus | Status |
|-----------|-------------|-----------|--------|
| 0–3 months | H1: Core quant stack | A1: Spec + data | ✅ Complete |
| 3–6 months | H2: Advanced risk | A2: Old Q1 (deprecated) | ✅ Complete |
| 6–9 months | **H3: Agent simulation core** | Empirical insurance track (lightweight) | **Next** |
| 9–12 months | **H4: Autonomous agent types** | — | Planned |
| 12–14 months | — | **A3: Emergent coordination** | Planned |
| 14–16 months | **H5: Population dynamics** | **A4: Regime manufacturing** | Planned |
| 16–18 months | — | **A5: Adversarial adaptation** | Planned |
| 18+ months | — | **A6: Retail impact & paper** | Planned |

**Parallelism:** H5 can begin with rule-based agents (from H3) before H4 is fully complete. The empirical insurance track runs independently. A3 starts after H4 delivers working autonomous agents.

---

## 6. Current Phase (March 2026)

- **HARBOR H1** (Core Quant Stack): ✅ Complete — data loaders, risk models, portfolio optimization, backtest engine all implemented and tested.
- **HARBOR H2** (Advanced Risk & Simulation): ✅ Complete — regime-aware covariance, Student-t/factor Monte Carlo, config-driven stress scenarios, risk decomposition, pluggable risk engine interface. 48 H2 tests passing.
- **ABF A2** (Old Q1 Pipeline): ✅ Complete, now deprecated — local projections, CAR computation, robustness sweep. Results were weak/time-dependent, motivating the pivot to autonomous-agent research.
- **ML Extensions**: Experimental scaffolding — LSTM/GRU vol forecasters and deep RL behavioral agents with unit tests. Classical baselines (GARCH, EWMA) implemented. These serve as infrastructure for RL agent types in H4.
- **Agent Simulation** (H3): **Next** — `harbor/agents/` stub in place. Building the market environment and agent interface is the immediate priority.

**Key scope change (March 2026):** Project pivoted from "AI-driven trading broadly" to "autonomous and agentic trading" specifically. Old Q1 deprecated. New three-question research arc: emergent coordination → regime manufacturing → adversarial adaptation. The simulation framework moved from late-stage (old H5) to immediate priority (new H3) as the primary research instrument. See `docs/PROJECT_EVOLUTION.md` for the full narrative and `docs/superpowers/specs/2026-03-16-autonomous-agent-pivot-design.md` for the design spec.

**Test suite:** 205 tests passing.
