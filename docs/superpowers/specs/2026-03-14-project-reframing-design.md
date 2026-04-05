# HARBOR Project Reframing — Design Spec

> **PRE-PIVOT** — Superseded by `2026-03-16-autonomous-agent-pivot-design.md`.

**Date:** 2026-03-14
**Author:** Lewis Smith
**Status:** Superseded

---

## 1. New Project Identity

**Updated description:** An asset management algorithm and empirical research system for studying how AI models and autonomous agents influence market behavior and shape outcomes for retail investors. It combines portfolio construction, risk management, and empirical analysis to measure how automated decision-making may create market patterns such as momentum, crowding, and instability.

**Core architecture — the feedback loop:**

1. **Research arm** (ABF) discovers how AI agents create market patterns (momentum, crowding, instability)
2. **Asset management arm** (HARBOR core) operationalizes those findings into risk rules and portfolio construction that mitigate those effects for retail investors
3. **Backtesting** validates that the mitigations protect retail portfolios
4. Results feed back into research refinement

This replaces the prior framing where HARBOR was "a portfolio tool" and ABF was "a research track." Instead: one system, two symbiotic arms, connected by a feedback loop.

---

## 2. Scope Changes

### 2.1 Broadened Agent Scope

**Before:** ML-driven trading strategies (trend-following, vol-targeting, systematic quant)
**After:** All AI models and autonomous agents — including LLM-based trading bots, robo-advisors, autonomous portfolio managers, and traditional systematic strategies

This is a deliberate broadening to reflect the evolving landscape where non-traditional AI (LLMs, multi-agent systems) enters financial markets.

### 2.2 Explicit Retail Investor Focus

**Before:** Retail focus was implicit (smaller portfolios, transparency, personal origin story)
**After:** Retail investor impact becomes an explicit, measurable part of the system with dedicated metrics and tooling

### 2.3 ABF Branding

ABF (Artificial Behavioral Finance) branding is retained in code identifiers (`harbor.abf`, configs, tests). The term "behavioral" encompasses agent behavior broadly. Documentation language is updated to reflect the broader scope.

---

## 3. Reframed Research Questions

### Q1: AI-Driven Momentum and Reversal

**Was:** Do ML-driven trend-following and volatility-targeting strategies create measurable medium-term autocorrelation regimes?

**Now:** Do AI-driven trading agents create measurable momentum and reversal patterns in asset prices?

**Mechanism:** When autonomous agents (vol-targeting algorithms, trend-followers, LLM-based bots) respond to the same market signals, their collective behavior can manufacture short-term momentum that subsequently reverses.

**Existing work preserved:** The shock→persistence→reversal pipeline (A1/A2) directly answers this under the broader framing.

### Q2: Agent Convergence and Instability

**Was:** Do ML agents trained on similar feature sets produce synchronized positioning that amplifies drawdowns?

**Now:** Does behavioral convergence among autonomous agents amplify drawdowns and destabilize correlations?

**Mechanism:** When multiple AI agents converge on similar strategies — whether through shared training data, similar architectures, or herding dynamics — their synchronized positioning can create correlation spikes and liquidity cascades. Broadens from "signal similarity" to include LLM-based convergence and robo-advisor herding.

### Q3: Signal Erosion from AI Proliferation

**Was:** Does widespread model-based trading reduce long-term alpha persistence?

**Now:** Does the proliferation of AI-driven strategies erode the effectiveness of established trading signals?

**Mechanism:** As more autonomous agents exploit the same factors, crowding compresses returns and accelerates factor decay. Broadens to include the effect of LLM-based agents entering markets at scale.

---

## 4. New Roadmap Phases

### HARBOR Framework

#### H5 — Agent Simulation Framework

**Module:** `harbor.agents`

**Depends on:** H3/H4 scaffolding validated (DRL agents working as baseline agent type)

**Goal:** Define, configure, and run heterogeneous AI agent populations in a market simulation environment. Agents could be:
- Rule-based (vol-targeting, momentum)
- ML-based (existing DRL agents from H3/H4)
- LLM-based (future — autonomous agents making allocation decisions)

The framework provides a market simulation environment where agent interactions generate synthetic order flow and price impact data. This enables causal testing of the patterns observed empirically in Q1/Q2.

**Exit criteria:** Configurable multi-agent simulation producing synthetic market data that can be analyzed with existing ABF pipelines.

**Research question mapping:** Supports Q1 (agent populations generating momentum/reversal), Q2 (convergence/crowding dynamics), and Q3 (signal erosion via crowding).

#### H6 — Retail Impact Analysis

**Module:** `harbor.retail`

**Goal:** Retail-specific metrics and tooling that make the retail investor focus tangible and measurable:
- Retail portfolio drawdown exposure during AI-driven regimes
- Cost-of-crowding estimates for typical retail allocations
- Comparison of regime-aware vs naive retail strategies
- Accessibility metrics (can a retail investor actually use this system's outputs?)

**Exit criteria:** Dashboard or report showing how AI-driven market patterns specifically affect retail portfolio outcomes, with before/after comparison using HARBOR's regime-aware mitigations.

### ABF Research

#### A4 — Agent-General Empirical Extensions

**Depends on:** H5 (agent simulation framework) producing usable synthetic data

**Goal:** Expand Q1/Q2/Q3 empirical work to incorporate agent simulation data from H5. Test whether synthetic agent populations reproduce the patterns found in real market data. This provides the causal link the current event-study approach can only suggest.

**Research question mapping:** Q1 (simulated momentum/reversal), Q2 (simulated crowding/correlation), Q3 (simulated signal erosion)

**Exit criteria:** Simulation-based evidence that agent convergence → momentum/crowding patterns, consistent with empirical findings from A2/A3.

#### A5 — Retail Impact Quantification

**Depends on:** H6 (retail impact analysis module)

**Goal:** Use H6 tools to measure how AI-driven market patterns specifically affect retail portfolios. Produce publication-quality figures like "retail drawdown amplification during crowding regimes."

**Exit criteria:** Working-paper section on retail impact with figures and statistical tests.

---

## 5. Updated Architecture

```
harbor/
  data/              Universe, price loaders, caching
  risk/              Covariance, HRP, Monte Carlo, regime detection, scenarios, decomposition
  portfolio/         Optimization, constraints, allocation
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters, behavioral RL agents
  abf/               ABF research experiment utilities (Q1, Q2)
  agents/            [NEW] Agent simulation framework
  retail/            [NEW] Retail impact metrics and analysis
notebooks/           Research and experimentation notebooks
experiments/         End-to-end scripts and prototypes
configs/             Config-driven experiment definitions
dashboard/           Portfolio monitoring dashboard
results/             Committed research outputs
```

Note: Only the `harbor/` package modules are shown in detail. Top-level directories (notebooks, experiments, configs, dashboard, results, docs, data, tests, research) are unchanged.

---

## 6. File Change Scope

### Modified files

#### `README.md` — Target structure:
1. Title + badge (keep)
2. **New tagline** (replace current "personal, research-driven..." quote with new description)
3. **Research Contribution** — rewrite to emphasize feedback loop and broader agent scope
4. **The Story** — keep existing narrative, add 1-2 paragraphs bridging to the AI agent vision
5. **What HARBOR Does** — restructured around two arms:
   - Asset Management (data, risk, portfolio, backtest)
   - Empirical Research (ABF Q1-Q3, agent simulation, retail impact)
6. **Research Track: ABF** — reframed Q1-Q3 with agent-general language
7. **Retail Investor Focus** — NEW section: why retail matters, what metrics/tooling will exist
8. **Repository Structure** — add `agents/` and `retail/` to the tree
9. Remaining sections (Status, Quickstart, Data, Docs, Disclaimer) — minor language updates

#### `docs/plan.md` — Target changes:
1. **Section 0 (Vision)** — rewrite with unified identity and feedback loop
2. **Section 1 (Architecture)** — add `harbor.agents` and `harbor.retail` to module list
3. **Sections 2.1-2.4** — keep existing H1-H4, mark H1/H2 as COMPLETE in headers
4. **Add Section 2.5** — H5 (Agent Simulation Framework) per Section 4 of this spec
5. **Add Section 2.6** — H6 (Retail Impact Analysis) per Section 4 of this spec
6. **Section 3.3-3.4** — keep existing A3/A4 text
7. **Add Section 3.5** — A4 reframed (Agent-General Extensions) per Section 4 of this spec
8. **Add Section 3.6** — A5 (Retail Impact Quantification) per Section 4 of this spec
9. **Section 5 (Milestone Table)** — extend with H5/H6/A4/A5 rows; mark completed phases with ✅
10. **Section 6 (Current Phase)** — update to reflect reframing

#### `docs/abf-prd.md` — Target changes:
1. **Executive Summary** — broaden from "systematic ML-driven trading strategies" to "AI models and autonomous agents"
2. **Research Objectives** — rewrite 3 objectives with agent-general language
3. **Q1 section** — reframe question, keep mechanism hypothesis but broaden agent types, keep test plan as-is
4. **Q2 section** — reframe question, broaden from "ML agents" to "autonomous agents", keep testing approach
5. **Q3 section** — reframe with AI proliferation language
6. **Add section** — "Retail Impact Integration" describing how findings connect to retail investor outcomes
7. **Connection to HARBOR section** — update to reference feedback loop and new modules

#### `CHANGELOG.md` — add entry under new version `v0.3.0-dev`:
```
## [0.3.0-dev] — 2026-03-14

### Changed
- Reframed project identity: unified asset management + empirical research system
- Broadened research scope from ML-driven strategies to all AI models and autonomous agents
- Reframed ABF Q1-Q3 research questions for agent-general scope
- Added explicit retail investor focus as measurable project goal

### Added
- `harbor.agents` module stub (Phase H5 — Agent Simulation Framework)
- `harbor.retail` module stub (Phase H6 — Retail Impact Analysis)
- New roadmap phases: H5, H6, A4 (agent-general extensions), A5 (retail impact)
```

### New files

| File | Purpose |
|------|---------|
| `harbor/agents/__init__.py` | Agent simulation framework stub |
| `harbor/retail/__init__.py` | Retail impact analysis stub |

**Stub content specification:**

`harbor/agents/__init__.py`:
```python
"""harbor.agents — Agent Simulation Framework

Defines, configures, and runs heterogeneous AI agent populations in a
market simulation environment. Agents interact to generate synthetic
order flow and price impact data for causal testing of ABF hypotheses.

Agent types (planned):
- Rule-based: vol-targeting, momentum, mean-reversion
- ML-based: DRL behavioral agents (from harbor.ml.behavior_agents)
- LLM-based: autonomous agents making allocation decisions

Status: Stub — implementation planned for Phase H5.
"""
```

`harbor/retail/__init__.py`:
```python
"""harbor.retail — Retail Impact Analysis

Metrics and tooling for measuring how AI-driven market patterns
affect retail investor portfolios. Operationalizes ABF research
findings into actionable retail-specific risk assessments.

Planned capabilities:
- Retail portfolio drawdown exposure during AI-driven regimes
- Cost-of-crowding estimates for typical retail allocations
- Regime-aware vs naive strategy comparison
- Retail-accessible risk reporting

Status: Stub — implementation planned for Phase H6.
"""
```

### Not touched

- All existing code modules (risk, portfolio, backtest, data, ml, abf)
- All tests (187 tests stay passing)
- All configs, experiments, notebooks
- ABF branding in code identifiers

---

## 7. Approach

**Evolutionary Expansion (Approach 1):** Keep the existing architecture and completed work intact. Layer new capabilities on top. Zero disruption to the 187 passing tests or completed phases. The new vision is an expansion of scope, not a contradiction of what exists.

---

## 8. Updated Milestone Table

| Timeframe | HARBOR Focus | ABF Focus | Status |
|-----------|-------------|-----------|--------|
| 0–3 months | H1: Data, HRP, Monte Carlo, backtest | A1: Spec + data, Q1 prep | ✅ Complete |
| 3–6 months | H2: Advanced risk, scenarios | A2: Q1 full execution + draft figures | ✅ Complete |
| 6–9 months | H3: NN volatility, DRL v1 | A3: Q2 proxies + baseline results | In progress (scaffolding done) |
| 9–12 months | H4: Behavioral agents integration | A3 continued | Scaffolding done, validation pending |
| 12–15 months | H5: Agent simulation framework | A4: Agent-general empirical extensions | Planned |
| 15–18 months | H6: Retail impact analysis | A5: Retail impact quantification | Planned |
| 18+ months | Polish, expand agent types (LLM) | Writing, external validation, publication | Planned |
