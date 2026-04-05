# Design Spec: Pivot to Autonomous & Agentic Trading

**Date:** 2026-03-16
**Author:** Lewis Smith
**Status:** Approved
**Supersedes:** `2026-03-15-phase2-agentic-trading-design.md` (old Phase 2 design with ML validation track)

**Phase label redefinition:** This spec redefines H3, H4, H5, A3-A6. The old meanings are deprecated:
| Label | Old meaning | New meaning |
|-------|------------|-------------|
| H3 | ML Validation (GARCH/EWMA benchmarking) | Agent Simulation Core |
| H4 | Behavioral Agent Validation (DRL tests) | Autonomous Agent Types (LLM, RL, HANGAR) |
| H5 | Agent Simulation Framework | Population Dynamics & Experiments |
| A3 | Q2 Crowding/Correlation (empirical) | Q1 Emergent Coordination (simulation + empirical) |

---

## 1. Summary

HANGAR pivots from studying "AI-driven trading broadly" to focusing specifically on **autonomous and agentic trading**. The thesis: autonomous agents (LLM-powered, RL-based, tool-using) create qualitatively different market dynamics than traditional systematic algos because they reason, adapt, and interact strategically.

ABF (Artificial Behavioral Finance) remains the research track. "A" still stands for "Artificial." The old Q1 (vol-shock persistence/reversal) is deprecated — its weak results (1.3bps) motivated the pivot. HANGAR core (data, risk, portfolio, backtest) stays as infrastructure and becomes a test participant: "how does a HANGAR-style portfolio perform when autonomous agents dominate the market?"

## 2. New Research Questions

Three-question arc, each building on the previous:

### Q1: Emergent Coordination
Do independently-built autonomous agents converge on similar strategies without explicit coordination? When multiple LLM/RL agents trade the same market, do they herd and crowd — even if trained/prompted independently?

### Q2: Regime Manufacturing
Do agent populations CREATE market regimes (momentum, mean-reversion, volatility clustering) that wouldn't exist without them? Agents don't just trade in regimes — they manufacture them through collective behavior.

### Q3: Adversarial Adaptation
Do autonomous agents learn to exploit each other? When agents detect other agents' strategies, does the market enter an adversarial arms race that destabilizes prices and harms retail investors?

## 3. Methodology

**Hybrid equal-weight:**
- **Simulation track:** Build autonomous agent populations, run experiments, generate synthetic market data
- **Empirical track:** Look for agent-like signatures in real post-2023 market data (unusual correlation patterns, herding metrics coinciding with LLM agent proliferation)
- **Validation:** Do simulation-generated patterns match what we see in real markets?

## 4. What Stays

- **H1** (Core Quant Stack): Complete, untouched
- **H2** (Advanced Risk & Simulation): Complete, untouched
- **Old Q1 code**: Removed — results documented in `docs/abf-q1-research-summary.md`
- **ML baselines** (`hangar/ml/volatility/baselines.py`): Infrastructure, still useful
- **All existing tests**: Continue passing

## 5. What's Deprecated

- Old A1/A2 (Q1 pipeline) — no longer on roadmap
- Old A3-A6 framing (crowding proxies, signal erosion)
- Old H3-H6 framing (ML validation → behavioral agents → late-stage simulation)
- Old Q1 research summary — gets deprecation note

## 6. New Phase Structure

### HANGAR Phases (Framework)

#### H3 — Agent Simulation Core
**Goal:** Build the market simulation environment and agent interface.

**Depends on:** H2

**Module:** `hangar/agents/`

```
hangar/agents/
    environment.py      # Market sim: price-impact model, order matching, state
    base_agent.py       # Abstract interface: observe() → decide() → act()
    rule_agents.py      # Momentum, vol-targeting, mean-reversion (simple baselines)
    config.py           # Population configs, parameter distributions
    metrics.py          # Market-level: crowding index, flow imbalance, regime labels
```

The environment uses a price-impact model (linear temporary + square-root permanent impact) — standard in agent-based finance literature.

**Exit criteria:**
- Market environment runs end-to-end with rule-based agents
- Price-impact model produces realistic price dynamics
- Metrics module computes crowding, flow imbalance, regime labels
- Output format compatible with ABF analysis pipelines
- Unit tests for environment, agents, metrics

#### H4 — Autonomous Agent Types
**Goal:** Implement the autonomous agent types that make this project distinctive.

**Depends on:** H3

```
hangar/agents/
    llm_agents.py       # LLM-based agents (Claude/GPT via API, prompted with market data)
    rl_agents.py        # RL agents (wraps hangar.ml.behavior_agents, trains in environment)
    hangar_agent.py     # HANGAR-as-agent: uses hangar.risk + hangar.portfolio to trade
    adaptation.py       # Agent learning/adaptation between rounds
```

Key design decisions:
| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM agent interface | API-based (Claude/GPT) | Real autonomous agents use LLMs; mocking loses the point. Development uses 5-10 agents with cached responses; final runs scale to 50 agents. |
| RL agents | Wrap existing hangar.ml.behavior_agents | Reuses existing code; treated as experimental baselines (not validated against classical benchmarks — old H3 Gate 2 was skipped in this pivot) |
| HANGAR-as-agent | Wraps hangar.risk + hangar.portfolio | Tests the system against itself |
| Adaptation | Between-round learning | Agents update strategy based on past performance |

**Exit criteria:**
- LLM agents receive market state, output trading decisions
- RL agents train within the simulation environment
- HANGAR agent uses existing portfolio logic to trade
- Adaptation mechanism allows strategy updates between rounds
- Unit tests for each agent type

#### H5 — Population Dynamics & Experiments
**Goal:** Multi-agent experiments and causal testing.

**Depends on:** H4

```
hangar/agents/
    population.py       # Population manager: spawn, configure, mix agent types
    experiments.py      # Predefined experiment configs
    analysis.py         # Bridge: convert sim output → ABF-compatible DataFrames
```

Experiment configurations:
1. Homogeneous LLM: 50 LLM agents, same base prompt → do they converge?
2. Mixed autonomous: 20 LLM + 20 RL + 10 HANGAR → interaction dynamics
3. Stress injection: External vol shock + mixed population → cascading behavior?
4. Adaptation test: Agents learn between rounds → convergence or divergence?
5. HANGAR resilience: HANGAR agent in autonomous-agent-dominated market

**Exit criteria:**
- Population manager configures and runs multi-agent simulations
- At least 3 experiment configurations run end-to-end
- Analysis module converts output to ABF-compatible format
- Results are reproducible via config files

### ABF Phases (Research)

#### A3 — Q1: Emergent Coordination
**Goal:** Test whether independently-built autonomous agents converge.

**Depends on:** H4

Method:
- Run agent populations with diverse initial configs (different LLM prompts, different RL seeds, different HANGAR configs)
- Measure: position correlation across agents over time, strategy similarity, herding intensity
- Key test: do agents that start different become more similar?
- Empirical complement: post-2023 signatures in real market data

**Exit criteria:**
- Cross-agent position correlation increases by >0.15 from initial to final simulation period (or null result documented with effect size)
- At least one convergence metric shows monotonic trend (Spearman ρ > 0.5 with simulation round) across 3+ independent runs
- Empirical comparison: cross-asset correlation structure pre-2020 vs post-2023 computed and compared
- Draft text for paper section

#### A4 — Q2: Regime Manufacturing
**Goal:** Test whether agent populations create market regimes.

**Depends on:** H5, A3

Method:
- Ablation: run simulation with and without agent populations, compare regime structure
- Dose-response: vary agent concentration 10% → 80%, measure regime intensity
- Regime detection: apply H2 tools to synthetic data
- Empirical complement: regime frequency pre-2020 vs post-2023

**Exit criteria:**
- Ablation: regime count or intensity differs at p<0.05 (permutation test) between agent vs no-agent simulations
- Dose-response: Spearman ρ > 0.7 between agent concentration and regime intensity
- At least one manufactured regime type identified with >20% higher occurrence than no-agent baseline
- Draft text for paper section

#### A5 — Q3: Adversarial Adaptation
**Goal:** Test whether agents learn to exploit each other.

**Depends on:** A4

Method:
- Multi-round simulation where agents adapt strategies between rounds
- Measure: price stability over time, strategy divergence/convergence, Sharpe decay
- Test: arms race vs equilibrium vs cycles
- Retail impact: naive buy-and-hold performance during adversarial periods

**Exit criteria:**
- Adaptation dynamics classified as one of: arms race (volatility increases >20% over rounds), equilibrium (volatility stabilizes within 10%), or cycles (significant autocorrelation in volatility at lag 3-10 rounds)
- Retail portfolio max drawdown quantified: at least 2 experiment configs compared
- HANGAR-as-agent outperformance (or not) vs naive buy-and-hold documented with Sharpe ratio comparison
- Draft text for paper section

#### A6 — Retail Impact & Write-up
**Goal:** Full paper draft with HANGAR-as-participant results.

**Depends on:** A5

Deliverables:
- HANGAR-as-agent performance across all experiment configs
- Quantified regime-awareness benefit for retail portfolios
- Paper-quality draft: Q1 + Q2 + Q3 + retail impact
- Reproducible pipeline via `make` targets

**Exit criteria:**
- Complete paper draft with figures, tables, statistical tests
- All experiments reproducible in <5 commands
- At least one expert review (faculty or practitioner)

## 7. Empirical Insurance Track

The simulation framework is the primary research instrument, but a lightweight empirical track runs in parallel as insurance. This reuses the crowding proxy methodology from the old A3 design (which depended only on H2, already complete).

**Scope:** Compute 2-3 crowding proxies (cross-sectional return dispersion, rolling factor-exposure correlation) on real S&P 500 data. Compare pre-2020 vs post-2023 structure. No causal claims — purely descriptive.

**Purpose:** If simulation takes longer than planned or LLM costs force scope reduction, this track produces real-data results that complement the simulation findings. It also strengthens the eventual paper by combining simulation evidence with real-market evidence.

**Effort:** ~2 weeks of work, can start anytime after H2 (now). Does not block any simulation phase.

## 8. LLM Agent Cost & Reproducibility Strategy

**Development:** 5-10 agents with cached API responses. Develop and debug using cached data to minimize costs.

**Intermediate runs:** Use cheaper/smaller models (e.g., Haiku) for statistical significance testing across multiple runs.

**Final runs:** Full API calls with production models (Claude Sonnet/Opus, GPT-4) for publication-quality results.

**Reproducibility:** All LLM responses cached with full request/response logging. Fixed random seeds for RL agents. Model version and temperature documented for each experiment run. Results tagged with model version so readers know exactly which LLM was used.

**Budget estimate:** ~$50-100 per full experiment run at 50 agents x 1000 timesteps. Development and intermediate runs: <$20 total with caching and smaller models.

## 9. Failure Contingencies

| Risk | Mitigation |
|------|-----------|
| LLM agents produce identical outputs (trivial convergence) | Vary temperature, prompts, and model providers. If still trivial, this is a finding: "LLM agents lack strategy diversity" is publishable. |
| Price-impact model produces unrealistic dynamics | Calibrate against historical S&P 500 volatility/autocorrelation. If calibration fails, simplify to proportional impact. |
| 50 agents computationally infeasible | Start with 5-10 agents. If scaling fails, document small-N results and note as limitation. |
| RL agents don't learn meaningful strategies | Use as experimental baselines only (acknowledged in spec). Focus research conclusions on LLM agents and rule-based agents. |
| Full simulation chain takes >6 months | Empirical insurance track (Section 7) provides fallback results. Publish simulation-only paper with whatever phases are complete. |

## 10. Milestone Table

| Timeframe | HANGAR Focus | ABF Focus | Status |
|-----------|-------------|-----------|--------|
| 0–3 months | H1: Core quant stack | A1: Spec + data | ✅ Complete |
| 3–6 months | H2: Advanced risk | A2: Old Q1 (deprecated) | ✅ Complete |
| 6–9 months | **H3: Agent simulation core** | Empirical insurance track (lightweight) | **Next** |
| 9–12 months | **H4: Autonomous agent types** | — | Planned |
| 12–14 months | — | **A3: Emergent coordination** | Planned |
| 14–16 months | **H5: Population dynamics** | **A4: Regime manufacturing** | Planned |
| 16–18 months | — | **A5: Adversarial adaptation** | Planned |
| 18+ months | — | **A6: Retail impact & paper** | Planned |

**Parallelism note:** H5 population dynamics can begin with rule-based agents (from H3) before H4 is fully complete. A3 starts after H4 delivers working autonomous agents. The empirical insurance track runs independently throughout.

## 11. Documentation Changes

| File | Change |
|------|--------|
| `docs/abf-prd.md` | Rewrite v3.0 — agent-first thesis, new Q1/Q2/Q3 |
| `docs/plan.md` | Rewrite phases H3-H5, A3-A6 |
| `docs/PROJECT_EVOLUTION.md` | New Chapter 5: The Pivot to Autonomous Agents |
| `docs/abf-q1-research-summary.md` | Add deprecation note |
| `CHANGELOG.md` | v0.4.0-dev entry |
| `MEMORY.md` | Update project state |

## 12. HANGAR's Triple Role

1. **Infrastructure:** Agents use HANGAR's risk models, portfolio construction, and data pipeline
2. **Participant:** HANGAR-as-agent competes against autonomous agents, testing regime-awareness
3. **Origin story:** The asset management system that sparked the research question
