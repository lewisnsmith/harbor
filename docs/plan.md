# HARBOR & ABF — Plan and Roadmap

Project Owner: Lewis Smith  
Last Updated: 2026-02-20  

---

## 0. Vision

HARBOR is an open-source **risk and portfolio framework** with deep mathematical underpinnings (HRP, Monte Carlo, risk decomposition) and **ML/RL/DL** extensions.  
ABF (Artificial Behavior in Financial Markets) is the research track that uses HARBOR to test whether systematic and ML-driven strategies manufacture autocorrelation regimes, crowding, and correlation spikes.

Outputs:
- A modular Python library (HARBOR) with serious quant / ML internals.
- A reproducible ABF research pipeline + working-paper-style draft.

---

## 1. Architecture Overview

High-level HARBOR modules:

- `harbor.data` – SP500 data, universe, features.
- `harbor.risk` – HRP, covariance estimation, Monte Carlo VaR/CVaR, regime detection.
- `harbor.portfolio` – optimization, constraints, allocation logic.
- `harbor.backtest` – engine, metrics, experiment runners.
- `harbor.ml.volatility` – NN volatility forecasters.
- `harbor.ml.behavior_agents` – deep RL behavioral portfolio agents.
- `harbor.abf` – Q1/Q2 experiments and analysis utilities.

ABF sits on top of this stack and uses it to run clean experiments.

---

## 2. HARBOR Roadmap (Framework)

### 2.1 Phase H1 — Core Quant Stack (0–3 months)

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
- One end-to-end script: “load SP500 → build HRP portfolio → backtest → report risk metrics”.

Phase H1 checklist (execution status):
- [x] `harbor.data`: point-in-time membership loader with documented fallback, price loader, risk-free loader, local cache scaffolding.
- [x] `harbor.risk`: sample/shrinkage covariance estimators, HRP implementation, Monte Carlo VaR/CVaR utilities.
- [x] `harbor.portfolio`: mean-variance, risk parity, and HRP allocation interfaces.
- [x] `harbor.backtest`: cross-sectional backtest engine with transaction costs and core metrics.
- [x] Exit-criteria script added: `experiments/h1_end_to_end_hrp_backtest.py`.
- [x] Phase H1 unit-test suite added for core data/risk/portfolio/backtest pathways.

---

### 2.2 Phase H2 — Advanced Risk & Simulation (3–6 months)

**Goal:** Deepen risk modeling and scenario analysis.

Deliverables:
- `harbor.risk`
  - Regime-aware covariance (e.g., regime switches or shrinkage based on vol state).
  - More robust Monte Carlo engines (non-Gaussian marginals; copula-based or factor models).
- Stress testing:
  - Shock scenarios (vol spikes, correlation spikes, sector crashes).
  - Risk decomposition by factor/cluster.

Exit criteria:
- Config-driven scenario runs with clear outputs (risk decomposition, stress-test reports).
- Pluggable risk engine used by ABF experiments.

---

### 2.3 Phase H3 — ML / DL Integration (6–9 months)

**Goal:** ML adds genuine capability, not just buzzwords.

**Status: Experimental scaffolding implemented ahead of schedule.** The modules below were built as exploratory work during H1/A2 development. They have unit tests but have **not** been validated against classical baselines or integrated into the production backtest pipeline. Formal validation (NN vol vs GARCH/EWMA, DRL vs buy-and-hold) is the gate for promoting these to production status.

Deliverables:
- `harbor.ml.volatility`
  - LSTM/GRU-based volatility forecasters with comparison to classical baselines.
  - Integration into vol-targeting and risk-parity strategies.
- `harbor.ml.behavior_agents` (v1)
  - Gym-style portfolio environment (SP500 subset).
  - One actor–critic DRL agent with a basic reward (risk-adjusted return).

Exit criteria:
- Experiments showing NN vol forecasts change risk outcomes (VaR/ES, realized volatility).
- DRL agent trained in the environment and evaluated in HARBOR backtests.

---

### 2.4 Phase H4 — Behavioral Agents & ABF Hooks (9–12 months)

**Goal:** HARBOR becomes a realistic platform for ABF.

**Status: Experimental scaffolding implemented ahead of schedule.** Behavioral reward shaping (loss aversion, overconfidence, return-chasing, disposition effect) and multi-agent simulation infrastructure are implemented with unit tests. Integration with ABF Q2 experiments is pending baseline validation from H3.

Deliverables:
- `harbor.ml.behavior_agents`
  - Behavioral reward shaping (loss aversion, overconfidence, return-chasing).
  - Multi-agent configurations (multiple agents with similar/different features).
- Tight integration with `harbor.abf`:
  - Agents can be used to generate “artificial” order/position data for ABF Q2 tests.

Exit criteria:
- Reproducible multi-agent experiments feeding into ABF crowding/correlation analyses.

---

## 3. ABF Roadmap (Research)

ABF = research layer that assumes HARBOR exists and is stable enough to run defined experiments.

### 3.1 Phase A1 — Spec & Data (0–2 months)

**Goal:** Lock questions, universe, and primary metrics.

Deliverables:
- PRD (`docs/abf-prd.md`) v1 done (already drafted; keep updated).
- SP500 universe with historical membership + clean daily panel.
- Config files for:
  - Shock definitions (Q1).
  - Correlation/crowding regimes (Q2).

Exit criteria:
- No more major changes to definitions of Q1/Q2 without version bump.
- Data loading and basic descriptive stats notebooks.

Phase A1 checklist (execution status):
- [x] PRD v1 maintained in `docs/abf-prd.md`.
- [x] Config scaffolding added: `configs/abf/q1_shock_definitions.json`, `configs/abf/q2_regime_definitions.json`.
- [x] Universe + panel loaders wired in `harbor.data`.
- [ ] Replace seed universe file with WRDS/CRSP full historical S&P 500 membership.
- [ ] Finalize descriptive-stats notebooks tied to the new loaders/configs.

---

### 3.2 Phase A2 — Q1: Shock → Persistence → Reversal (2–5 months)

**Goal:** Fully execute ABF Question 1.

Deliverables:
- `harbor.abf.q1`:
  - Functions to identify shocks, build event windows, compute persistence/reversal metrics.
  - Local projection / event-study regressions with vol-control proxies.
- Notebooks:
  - `abf_q1_shock_definition.ipynb`
  - `abf_q1_persistence_reversal.ipynb`

Exit criteria:
- Stable figures/tables answering Q1 with documented robustness checks.
- Code path from raw data → final figures is reproducible in < 5 commands.

---

### 3.3 Phase A3 — Q2: Similarity, Crowding, Correlation (5–9 months)

**Goal:** Execute ABF Question 2 with at least one strong proxy family.

Deliverables:
- `harbor.abf.q2`:
  - Signal similarity and crowding proxy computation.
  - Correlation spike detection.
  - Regressions linking proxies to correlation/drawdown behavior.
- Integration with DRL / ML agents (optional but desirable):
  - Use behavior agents and ML signals as additional “agents” in the analysis.

Exit criteria:
- Clear evidence (or null results) on whether similarity/crowding proxies predict correlation spikes and drawdown amplification.
- Draft text for a paper section on Q2.

---

### 3.4 Phase A4 — Writing, Validation, and External Feedback (9–12+ months)

**Goal:** Turn ABF work into a credible research artifact.

Deliverables:
- Draft ABF paper (Q1 + Q2).
- Public or semi-public repo with:
  - `docs/` and `notebooks/` aligned to the paper.
  - Reproducible instructions.
- External validation:
  - At least one faculty member or practitioner has reviewed and commented.
  - Potential reading group / independent study based on ABF.

Exit criteria:
- You can send a concise ABF writeup + repo link to Stanford/MIT apps as a serious project.
- You have at least one expert willing to reference the work.

---

## 4. Cross-Cutting Practices

To keep the project authentic and professional:

- **Versioning & Changelog**
  - Semantic Versioning: `MAJOR.MINOR.PATCH`.
  - `CHANGELOG.md` updated for each tagged release (0.1.0 = core HARBOR; 0.2.0 = Q1 baseline; etc.). [web:141][web:139][web:146]

- **Documentation**
  - `README.md` for overview and quickstart.
  - `docs/abf-prd.md` for research spec.
  - `plan.md` (this file) for roadmap and phases.

- **Git Hygiene**
  - Feature branches per module/experiment.
  - Descriptive, technical commit messages and PRs.

- **Reproducibility**
  - Config-driven experiments.
  - Minimal “one-command” scripts to reproduce key results.

---

## 5. Milestone Table (High-Level)

| Timeframe             | HARBOR Focus                         | ABF Focus                               |
|-----------------------|---------------------------------------|-----------------------------------------|
| 0–3 months            | Data, HRP, Monte Carlo, backtest     | Spec + data, Q1 prep                    |
| 3–6 months            | Advanced risk, scenarios             | Q1 full execution + draft figures       |
| 6–9 months            | NN volatility, DRL v1                | Q2 proxies + baseline results           |
| 9–12+ months          | Behavioral agents integration        | Writing, external validation, polishing |

---

## 6. Current Phase (March 2026)

- **HARBOR H1** (Core Quant Stack): Complete — data loaders, risk models, portfolio optimization, backtest engine all implemented and tested.
- **HARBOR H2** (Advanced Risk & Simulation): Complete — regime-aware covariance, Student-t/factor Monte Carlo, config-driven stress scenarios, risk decomposition, pluggable risk engine interface. 48 H2 tests passing. Demo script at `experiments/h2_risk_engine_demo.py`.
- **ABF A1/A2** (Q1 Pipeline): Analysis pipeline complete — local projections, CAR computation, robustness sweep, figure generation. Preliminary results committed to `results/abf_q1/`. Experiments README updated with reproducible commands.
- **ML Extensions** (H3/H4): Experimental scaffolding — LSTM/GRU vol forecasters and deep RL behavioral agents implemented with unit tests. Not yet validated against classical baselines. Formal validation is the next milestone.
- **ABF A3** (Q2 — Crowding/Correlation): Stub only — `harbor.abf.q2` config scaffolding in place, implementation pending H3 validation.
