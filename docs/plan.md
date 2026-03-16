# HARBOR & ABF — Plan and Roadmap

Project Owner: Lewis Smith
Last Updated: 2026-03-15

---

## 0. Vision

HARBOR is an asset management algorithm and empirical research system for studying how AI models and autonomous agents influence market behavior and shape outcomes for retail investors. It combines portfolio construction, risk management, and empirical analysis to measure how automated decision-making may create market patterns such as momentum, crowding, and instability.

**Core architecture — the feedback loop:**

1. **Research arm** (ABF) discovers how AI agents create market patterns (momentum, crowding, instability)
2. **Asset management arm** (HARBOR core) operationalizes those findings into risk rules and portfolio construction that mitigate those effects for retail investors
3. **Backtesting** validates that the mitigations protect retail portfolios
4. Results feed back into research refinement

ABF (Artificial Behavioral Finance) is the research layer that uses HARBOR to run clean, reproducible experiments testing whether AI-driven trading behavior manufactures autocorrelation regimes, crowding, and correlation spikes.

Outputs:
- A modular Python library (HARBOR) with serious quant / ML / agent simulation internals.
- A reproducible ABF research pipeline + working-paper-style draft.

---

## 1. Architecture Overview

High-level HARBOR modules:

- `harbor.data` – SP500 data, universe, features, Massive API client.
- `harbor.risk` – HRP, covariance estimation, Monte Carlo VaR/CVaR, regime detection, scenarios, decomposition.
- `harbor.portfolio` – optimization, constraints, allocation logic.
- `harbor.backtest` – engine, metrics, experiment runners.
- `harbor.ml.volatility` – NN volatility forecasters (experimental).
- `harbor.ml.behavior_agents` – deep RL behavioral portfolio agents (experimental).
- `harbor.abf` – Q1/Q2/agent validation experiments and analysis utilities.
- `harbor.agents` – Agent simulation framework (planned).
- `harbor.retail` – Retail impact metrics and analysis (planned).

The system has two arms: the asset management stack (data → risk → portfolio → backtest) and the research stack (abf + agents + retail). They share infrastructure and feed results into each other.

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

### 2.3 Phase H3 — ML Validation (6–8 months)

**Goal:** Promote experimental ML scaffolding to validated status by benchmarking against classical baselines.

**Status: Scaffolding implemented, validation pending.** LSTM/GRU vol forecasters and DRL behavioral agents exist with unit tests but have not been benchmarked against classical baselines.

**Validation Gates:**

**Gate 1 — NN Volatility Forecasters:**
- Benchmark LSTM/GRU vol forecasters against GARCH(1,1) and EWMA baselines.
- Metrics: RMSE, MAE, QLIKE, directional accuracy.
- Walk-forward validation on rolling out-of-sample windows.
- Pass criterion: NN beats at least one classical baseline on 2+ of 4 metrics across 2+ test windows.
- If fail: document result, keep module as experimental, proceed with classical vol models.

**Gate 2 — DRL Agent Baseline:**
- Benchmark actor-critic DRL agent against buy-and-hold and equal-weight baselines.
- Metrics: Sharpe ratio, max drawdown, turnover.
- Pass criterion: DRL agent achieves risk-adjusted return within 80% of buy-and-hold.
- If fail: simplify reward function, reduce action space, document learnings.

Deliverables:
- `harbor.ml.volatility`
  - LSTM/GRU-based volatility forecasters with documented comparison to GARCH/EWMA.
  - Integration into vol-targeting and risk-parity strategies (if Gate 1 passes).
- `harbor.ml.behavior_agents` (v1)
  - Gym-style portfolio environment (SP500 subset).
  - Actor-critic DRL agent benchmarked and documented.

Exit criteria:
- Gate 1 and Gate 2 evaluated with documented results (pass or fail).
- Validated models integrated into `harbor.risk` vol forecasting pipeline.
- Results notebook committed with reproducible commands.

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H3-S1 | 2 weeks | Classical baselines (GARCH, EWMA), walk-forward eval harness |
| H3-S2 | 2 weeks | NN vol benchmark runs, DRL agent benchmark, Gate 1+2 decisions |
| H3-S3 | 2 weeks | Integration: plug validated forecasters into risk engine, document results |

---

### 2.4 Phase H4 — Behavioral Agent Validation (8–10 months)

**Goal:** Validate that behavioral reward shaping produces meaningfully different agent behavior, and integrate agents into the backtest pipeline.

**Status: Scaffolding implemented, validation pending.** Behavioral reward shaping (loss aversion, overconfidence, return-chasing, disposition effect) and multi-agent infrastructure exist with unit tests. Validation depends on H3 Gate 2.

**Depends on:** H3 Gate 2 passed (or simplified DRL agent working).

Deliverables:
- `harbor.ml.behavior_agents`
  - Behavioral differentiation tests: each reward type vs neutral baseline.
  - Metrics: position concentration, turnover, drawdown profile, holding period distribution.
  - Pass criterion: 3+ of 4 behavioral types produce statistically distinguishable characteristics (KS test, p<0.05).
  - Multi-agent configurations (2-5 agents, mixed behavioral types).
- Backtest integration:
  - Agents as "strategy generators" in the backtest engine.
  - Agent-generated portfolios evaluated with standard HARBOR metrics.

Exit criteria:
- 3+ behavioral types produce distinguishable behavior.
- Multi-agent config runs without errors.
- Agent portfolios evaluable through standard backtest pipeline.

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H4-S1 | 2 weeks | Behavioral differentiation tests, per-agent metrics |
| H4-S2 | 2 weeks | Multi-agent config, backtest integration, exit criteria |

---

### 2.5 Phase H5 — Agent Simulation Framework (10–13 months)

**Goal:** Build a market simulation environment where heterogeneous AI agent populations interact, generating synthetic order flow and price impact data analyzable with the same ABF pipelines used on real data.

**Depends on:** H4 (validated behavioral agents as one agent type).

**Module:** `harbor.agents`

Architecture:
```
harbor/agents/
    environment.py      # Market simulation environment (price-impact model)
    base_agent.py       # Abstract agent interface (observe/act/update)
    rule_agents.py      # Vol-targeting, momentum, mean-reversion
    ml_agents.py        # Wrapper around harbor.ml.behavior_agents
    llm_agents.py       # LLM-based agent (stub — Phase 3 scope)
    population.py       # Agent population manager + config loader
    metrics.py          # Crowding index, flow imbalance, synthetic vol-control proxy
```

Agent types at launch:
| Type | Source | Behavior |
|------|--------|----------|
| Rule-based | `rule_agents.py` | Momentum, vol-targeting, mean-reversion with configurable parameters |
| ML-based | `ml_agents.py` | Wraps validated DRL agents from H4 |
| LLM-based | `llm_agents.py` | Stub — interface defined, `NotImplementedError`. Phase 3 scope. |

Key design decisions:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Price formation | Impact model (not order book) | Captures dynamics we care about; standard in agent-based finance lit |
| Agent heterogeneity | Configurable parameter distributions | Same type with different params creates realistic diversity |
| LLM agents | Stub only in Phase 2 | Fascinating but scope-risky; interface defined now |
| Output format | Same DataFrame structure as real data | Enables reuse of entire ABF analysis pipeline |

Experiment configurations (planned):
1. Homogeneous momentum: 50 momentum agents → does crowding emerge?
2. Mixed population: 20 momentum + 20 vol-target + 10 mean-reversion → correlation dynamics
3. Stress injection: external vol shock + mixed population → cascading deleveraging?
4. Convergence test: diverse parameters, agents learn → do they converge?

Exit criteria:
- Configurable multi-agent simulation runs end-to-end.
- Rule-based + ML-based agents produce synthetic price/volume data.
- Metrics module computes crowding, flow imbalance, synthetic vol-control proxy.
- Output format compatible with ABF analysis pipelines.
- At least 2 experiment configurations produce non-trivial results.

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| H5-S1 | 2 weeks | Market environment + base agent interface + rule-based agents |
| H5-S2 | 2 weeks | Population manager, ML agent wrapper, simulation loop |
| H5-S3 | 2 weeks | Metrics module, experiment configs, end-to-end simulation runs |

---

### 2.6 Phase H6 — Retail Impact Analysis (15–18 months)

**Goal:** Retail-specific metrics and tooling that make the retail investor focus tangible and measurable.

**Module:** `harbor.retail`

Deliverables:
- Retail portfolio drawdown exposure during AI-driven regimes.
- Cost-of-crowding estimates for typical retail allocations.
- Comparison of regime-aware vs naive retail strategies.
- Accessibility metrics (can a retail investor actually use this system's outputs?).

Exit criteria:
- Dashboard or report showing how AI-driven market patterns specifically affect retail portfolio outcomes, with before/after comparison using HARBOR's regime-aware mitigations.

---

## 3. ABF Roadmap (Research)

ABF = research layer that assumes HARBOR exists and is stable enough to run defined experiments.

### 3.1 Phase A1 — Spec & Data ✅ COMPLETE

**Goal:** Lock questions, universe, and primary metrics.

Deliverables:
- PRD (`docs/abf-prd.md`) v1 done.
- SP500 universe with historical membership + clean daily panel.
- Config files for shock definitions (Q1) and correlation/crowding regimes (Q2).

Phase A1 checklist (execution status):
- [x] PRD v1 maintained in `docs/abf-prd.md`.
- [x] Config scaffolding added: `configs/abf/q1_shock_definitions.json`, `configs/abf/q2_regime_definitions.json`.
- [x] Universe + panel loaders wired in `harbor.data`.
- [ ] Replace seed universe file with WRDS/CRSP full historical S&P 500 membership (Massive API integration designed, pending implementation).
- [ ] Finalize descriptive-stats notebooks tied to the new loaders/configs.

---

### 3.2 Phase A2 — Q1: Shock → Persistence → Reversal ✅ COMPLETE

**Goal:** Fully execute ABF Question 1 — do AI-driven trading agents create measurable momentum and reversal patterns?

Deliverables:
- `harbor.abf.q1`:
  - Shock detection, event windows, persistence/reversal metrics.
  - Local projection / event-study regressions with vol-control proxies.
  - Robustness sweep and automated figure generation.
- Results committed to `results/abf_q1/`.

Exit criteria:
- Stable figures/tables answering Q1 with documented robustness checks.
- Code path from raw data → final figures is reproducible in < 5 commands.

---

### 3.3 Phase A3 — Q2: Agent Convergence & Instability (6–10 months)

**Goal:** Empirically test whether behavioral convergence among autonomous agents amplifies drawdowns and destabilizes correlations — using real market data.

**Runs in parallel with H3/H4 (Track 1 of two-track Phase 2 structure). Depends only on H2 (complete).**

**Module:** `harbor.abf.q2`

Proxy construction:
| Proxy | Definition |
|-------|-----------|
| Momentum crowding | Cross-sectional return dispersion (low dispersion = crowded positioning) |
| Signal similarity | Rolling correlation of factor exposures across sectors/size groups |
| ETF flow pressure | Aggregate ETF inflow/outflow as systematic activity measure |
| Vol-control exposure | Risk-parity deleveraging indicator (from H2) |

Detection:
- Correlation spike detector: periods where realized cross-asset correlation exceeds trailing 75th percentile.
- Drawdown severity classifier: regime labels for "normal" vs "amplified" drawdown.

Regressions:
- Crowding proxies → correlation expansion (predictive, 1-5 day lead).
- Crowding proxies → drawdown severity (controlling for baseline vol, liquidity).
- Granger causality tests: does crowding lead correlation spikes?

Target figures:
1. Crowding proxy time series with correlation spike overlay.
2. Drawdown severity conditional on crowding regime.
3. Lead-lag heatmap: crowding proxy → correlation expansion at various horizons.
4. Robustness: pre/post-2020, high/low liquidity splits.

Exit criteria:
- At least one crowding proxy predicts correlation expansion out-of-sample (AUC >0.65).
- Granger causality at 1-5 day lead with p<0.05.
- Robustness to 2 sub-sample splits.
- Reproducible in <5 commands.
- Draft text for a paper section on Q2.

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| A3-S1 | 2 weeks | Proxy construction + correlation spike detector |
| A3-S2 | 2 weeks | Regressions + Granger causality |
| A3-S3 | 2 weeks | Figures, robustness checks |
| A3-S4 | 2 weeks | Write-up, reproducibility, integration with risk engine |

---

### 3.4 Phase A4 — Agent-General Empirical Extensions (13–15 months)

**Goal:** Compare synthetic agent simulation data (H5) against real market patterns (A3). The central question: do simulated agent populations, with no access to real market data, independently produce the same statistical signatures found empirically?

**Depends on:** A3 and H5 both complete. This is the convergence point of the two-track Phase 2 structure.

**Module:** `harbor.abf.agent_validation`

Pattern matching framework:
- Compute exact A3 metrics on H5's synthetic data.
- Distribution comparison: KS test, Wasserstein distance between synthetic and real metric distributions.
- Sensitivity analysis: which simulation parameters most affect pattern reproduction?

Causal experiments (impossible with real markets — simulation's unique contribution):
- **Ablation:** Remove vol-targeting agents → do correlation spikes disappear?
- **Dose-response:** Increase momentum agent proportion 10% → 80% → does crowding increase monotonically?
- **Heterogeneity test:** High-diversity vs low-diversity populations → does convergence produce worst outcomes?

Target figures:
1. Side-by-side: real vs synthetic crowding proxy distributions.
2. Ablation results: which agent types drive which patterns.
3. Dose-response curves: agent concentration → crowding severity.
4. Phase diagram: agent diversity x market stress → outcome severity.

Exit criteria:
- At least one synthetic metric distribution statistically consistent with real data (KS p>0.05).
- Ablation shows at least one agent type whose removal significantly changes a pattern.
- Results in paper-quality figures with reproducible commands.
- Draft text for "simulation validates empirical findings" paper section.

Sprint plan:
| Sprint | Duration | Focus |
|--------|----------|-------|
| A4-S1 | 2 weeks | Pattern matching framework, metric computation on synthetic data |
| A4-S2 | 2 weeks | Causal experiments (ablation, dose-response) |
| A4-S3 | 2 weeks | Figures, statistical comparison, write-up |

---

### 3.5 Phase A5 — Retail Impact Quantification (15–18 months)

**Goal:** Use H6 tools to measure how AI-driven market patterns specifically affect retail portfolios.

**Depends on:** H6 (retail impact analysis module).

Deliverables:
- Publication-quality figures: "retail drawdown amplification during crowding regimes."
- Statistical tests comparing retail portfolio outcomes in AI-active vs baseline regimes.
- Working-paper section on retail impact.

Exit criteria:
- Paper section with figures and statistical tests ready for inclusion in ABF draft.

---

### 3.6 Phase A6 — Writing, Validation, and External Feedback (18+ months)

**Goal:** Turn ABF work into a credible research artifact.

Deliverables:
- Draft ABF paper (Q1 + Q2 + agent simulation + retail impact).
- Public or semi-public repo with docs and notebooks aligned to the paper.
- External validation: at least one faculty member or practitioner has reviewed and commented.

Exit criteria:
- Concise ABF writeup + repo link suitable for Stanford/MIT applications.
- At least one expert willing to reference the work.

---

## 4. Cross-Cutting Practices

To keep the project authentic and professional:

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
| 0–3 months | H1: Data, HRP, Monte Carlo, backtest | A1: Spec + data, Q1 prep | ✅ Complete |
| 3–6 months | H2: Advanced risk, scenarios | A2: Q1 full execution + draft figures | ✅ Complete |
| 6–8 months | H3: ML validation (NN vol, DRL) | A3: Q2 proxies + crowding regressions | **Next** |
| 8–10 months | H4: Behavioral agent validation | A3: Q2 figures, robustness, write-up | Planned |
| 10–13 months | H5: Agent simulation framework | A4: Pattern matching (sim vs real) | Planned |
| 13–15 months | H5: Agent experiments | A4: Causal experiments + write-up | Planned |
| 15–18 months | H6: Retail impact analysis | A5: Retail impact quantification | Planned |
| 18+ months | LLM agents, polish | A6: Writing, external validation, publication | Planned |

**Phase 2 execution structure:** Two parallel tracks converging at A4.
- Track 1 (Empirical): A3 runs immediately (depends on H2, complete).
- Track 2 (Simulation): H3 → H4 → H5 (sequential dependency chain).
- Convergence: A4 compares simulation output against A3's real-data findings.

---

## 6. Current Phase (March 2026)

- **HARBOR H1** (Core Quant Stack): ✅ Complete — data loaders, risk models, portfolio optimization, backtest engine all implemented and tested.
- **HARBOR H2** (Advanced Risk & Simulation): ✅ Complete — regime-aware covariance, Student-t/factor Monte Carlo, config-driven stress scenarios, risk decomposition, pluggable risk engine interface. 48 H2 tests passing. Demo script at `experiments/h2_risk_engine_demo.py`.
- **ABF A1/A2** (Q1 Pipeline): ✅ Complete — local projections, CAR computation, robustness sweep, figure generation. Preliminary results committed to `results/abf_q1/`. Experiments README updated with reproducible commands.
- **ML Extensions** (H3/H4): Experimental scaffolding — LSTM/GRU vol forecasters and deep RL behavioral agents implemented with unit tests. Not yet validated against classical baselines. Formal validation is the next milestone.
- **ABF A3** (Q2 — Crowding/Correlation): Starting — `harbor.abf.q2` config scaffolding in place, proxy construction and regressions are next. Runs in parallel with H3 (Track 1 of two-track structure).
- **Agent Simulation** (H5): Planned — `harbor/agents/` stub in place. Depends on H4 validation.
- **Retail Impact** (H6): Planned — `harbor/retail/` stub in place. Phase 3 scope.

**Test suite:** 187 tests passing, targeting ~290 by end of Phase 2.

**Key scope change (March 2026):** Project reframed from "ML-driven trading strategies" to "all AI models and autonomous agents." Research questions broadened to cover LLM-based trading bots, robo-advisors, and autonomous portfolio managers alongside traditional systematic strategies. See `docs/PROJECT_EVOLUTION.md` for full narrative and `docs/superpowers/specs/2026-03-14-project-reframing-design.md` for the design spec.
