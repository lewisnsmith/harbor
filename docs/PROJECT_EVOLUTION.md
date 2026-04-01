# HARBOR — Project Evolution

**Author:** Lewis Smith
**Started:** October 2025
**Last updated:** March 15, 2026

This document tells the story of how HARBOR evolved from a competition entry into an empirical research system studying how AI agents reshape financial markets. Each chapter captures what was built, what was learned, and how that learning changed the project's direction.

---

## Chapter 1: The Builder (October 2025 – January 2026)

### Origin

HARBOR began as a portfolio management system built for the Wharton Global Investment Competition. The initial goal was straightforward: build a system that could construct, backtest, and manage a multi-asset portfolio with real risk controls.

### What Was Built

- **v0.1.0-dev** (February 2026 release, but reflecting work from Oct 2025 onward)
- `harbor.data`: S&P 500 universe loaders with survivorship-bias-aware fallbacks, chunked concurrent price fetching, risk-free rate proxy, local Parquet/pickle cache
- `harbor.risk`: Sample and Ledoit-Wolf shrinkage covariance estimators, Hierarchical Risk Parity (HRP) allocation, Monte Carlo VaR/CVaR simulation, regime detection (vol shocks, correlation spikes)
- `harbor.portfolio`: Mean-variance, risk parity, and HRP weight interfaces with regime-aware position sizing
- `harbor.backtest`: Sharpe, Sortino, Calmar, max drawdown, and win rate metrics
- CI pipeline with ruff linting and pytest

### Key Technical Decisions

- **HRP over mean-variance as default:** Mean-variance is elegant but notoriously unstable with estimated inputs. HRP uses hierarchical clustering on the correlation matrix to build portfolios that don't invert covariance matrices. This was the first decision that pushed me deeper into the math — understanding why HRP works requires understanding eigenvector stability, dendrogram structure, and the bias-variance tradeoff in covariance estimation.
- **Survivorship bias awareness from day one:** Used historical S&P 500 membership rather than today's constituents. This is a mistake most student projects make, and avoiding it early kept the research foundation clean.
- **Config-driven experiments:** Every experiment parameterized through JSON configs, not hardcoded. This paid off enormously in later phases when running robustness sweeps.

### What Was Learned

Building the risk models forced me to confront a deeper question: these models assume market structure is exogenous — that correlations and volatilities are properties of the assets themselves. But the models I was building (vol-targeting, risk parity) are the same models institutional investors use. If enough participants run the same math, do they *change* the correlations and volatilities they're measuring?

This question wouldn't leave me alone.

---

## Chapter 2: The Researcher (February 2026 – Early March 2026)

### The Shift

The question from Chapter 1 became a formal research agenda. I wrote a PRD (Product Requirements Document) defining "Artificial Behavioral Finance" (ABF) — a framework for studying how systematic ML-driven trading strategies create endogenous market dynamics.

The core thesis: when institutional participants deploy similar trend-following and volatility-targeting models at scale, they manufacture autocorrelation regimes, synchronized positioning, and amplified drawdowns that traditional behavioral finance cannot explain because the "agents" are algorithms, not humans.

### What Was Built

- **v0.2.0-dev** (February 27, 2026)
- **ABF Q1 pipeline** (`harbor.abf.q1`): Complete analysis pipeline for testing whether vol-targeting behavior creates momentum persistence and reversal patterns
  - Shock detection and event window construction
  - Local projection regressions with Newey-West HAC standard errors
  - Cumulative abnormal return computation
  - Robustness sweep across alternative shock definitions, sub-sample splits, and lag sensitivities
  - Automated figure generation
- **Advanced risk engine** (Phase H2): Regime-aware covariance estimation, Student-t and factor-driven Monte Carlo engines, config-driven stress scenarios (vol spikes, correlation spikes, sector crashes), risk decomposition by factor and cluster
- **ML scaffolding** (experimental): LSTM/GRU volatility forecasters, deep RL behavioral agents with behavioral reward shaping (loss aversion, overconfidence, return-chasing, disposition effect), multi-agent simulation infrastructure
- **187 passing tests** across all modules

### Key Technical Decisions

- **Event-study methodology over time-series regression:** Local projections allow impulse response estimation at multiple horizons without the stationarity assumptions of VARs. This was influenced by Jorda (2005) and aligned with the shock → persistence → reversal hypothesis structure.
- **Newey-West HAC standard errors:** Financial return data has well-documented autocorrelation and heteroskedasticity. Using OLS standard errors would overstate significance. HAC errors are conservative but honest.
- **ML as scaffolding, not claims:** Built LSTM vol forecasters and DRL agents but explicitly labeled them as experimental — not validated against classical baselines. The discipline of not claiming results before validation is important.

### UCLA Faculty Outreach

Identified target faculty at UCLA Anderson (Avanidhar Subrahmanyam, Shlomo Benartzi) and IPAM for potential collaboration. Prepared 1-page research memo with the problem statement and preliminary Q1 results.

### What Was Learned

Two things became clear during this phase:

1. **The empirical evidence alone can only suggest, not prove, causation.** Event studies show that shocks followed by vol-control pressure lead to momentum persistence and reversal — but this could be correlation, not causation. To make the causal argument, I would need a simulation environment where I can control agent behavior and test counterfactuals.

2. **The landscape was shifting underneath me.** By early March 2026, LLM-based trading bots, autonomous portfolio managers, and agentic AI systems were entering financial markets in ways that my original "ML-driven trend-following" framing didn't capture. The research questions were right, but the scope was too narrow.

---

## Chapter 3: The Reframing (March 2026)

### Why It Happened

Three developments converged:

1. **Agentic AI explosion:** LLM-based autonomous agents were being deployed for portfolio management, trade execution, and market analysis. These agents behave differently from traditional systematic strategies — they can reason, adapt, and potentially herd in novel ways.

2. **The causal gap:** The Q1 empirical results showed patterns consistent with manufactured autocorrelation, but couldn't isolate agent behavior as the cause. A simulation framework became necessary, not optional.

3. **A retail dimension:** As AI agents proliferate, retail investors are the most exposed — they lack the tools to detect or respond to AI-driven dynamics. This added a real-world motivation, but it was secondary. The simulation-first methodology was already necessary to answer the research questions regardless of application.

### What Changed

- **Project identity:** From "a portfolio tool with a research track" to "an integrated research system where empirical findings inform risk mitigations and simulations test causal mechanisms"
- **Agent scope:** From "ML-driven trend-following and volatility-targeting" to "all AI models and autonomous agents — including LLM-based trading bots, robo-advisors, and autonomous portfolio managers"
- **Research questions reframed:**
  - Q1: "Do AI-driven trading agents create measurable momentum and reversal patterns?"
  - Q2: "Does behavioral convergence among autonomous agents amplify drawdowns and destabilize correlations?"
  - Q3: "Does the proliferation of AI-driven strategies erode the effectiveness of established trading signals?"
- **New modules planned:** `harbor.agents` (agent simulation framework), `harbor.retail` (retail impact analysis)

### The Design Decision

I chose an evolutionary expansion approach — layer new capabilities on top of the existing, validated architecture rather than rewriting. The 187 passing tests and completed phases (H1, H2, A1, A2) remain untouched. This is the software engineering equivalent of "don't break what works."

### What Was Learned

The most important lesson from the reframing: **a good project adapts to the world, it doesn't just execute a static plan.** The original ABF thesis was correct in mechanism but narrow in scope. Broadening it didn't invalidate the work — it made it more relevant. The existing Q1 pipeline answers the reframed Q1 just as well, because the underlying mechanism (algorithmic behavior → momentum → reversal) is the same regardless of whether the algorithm is a vol-targeting model or an LLM-based agent.

---

## Chapter 4: The Brief Scientist (March 14–15, 2026)

### What Happened

After the reframing, I designed a two-track parallel research structure: Track 1 (empirical crowding analysis on real data) running alongside Track 2 (ML validation → behavioral agents → agent simulation). The convergence point would compare simulation output against real-market findings.

I also implemented classical volatility forecasting baselines (GARCH(1,1), EWMA) to benchmark the experimental ML models, fixed a multicollinearity bug in the Q1 analysis (day-of-week dummies + constant = perfect collinearity, causing NaN p-values), and brought the test suite to 205 passing tests.

### What Was Learned

Two things became clear almost immediately:

1. **The old Q1 results were a dead end.** Even after fixing the multicollinearity bug, the effects were economically tiny (~1.3bps). Traditional vol-targeting algorithms produce small effects because they're simple — they follow rules mechanically. The interesting question was never about simple algorithms.

2. **Autonomous agents are qualitatively different.** LLM-based trading agents don't just follow rules — they reason, adapt, and interact strategically. An LLM agent can read market commentary, infer other agents' positions, and adjust. An RL agent can learn to exploit patterns created by other agents. These are not just "smarter algorithms" — they're a different category of market participant.

This realization made the two-track structure feel misaligned. Track 1 (empirical crowding on real data) was still framed around traditional algorithm effects. The simulation framework was buried as a late-stage addition (old H5). But the simulation is the point — it's the only way to study autonomous agent dynamics with causal rigor.

---

## Chapter 5: The Pivot to Autonomous Agents (March 16, 2026 – Present)

### Why It Happened

The progression was fast but logical:

1. **Built the system** (H1/H2) → learned how institutional algorithms work
2. **Tested the obvious question** (old Q1) → found the effects are too small to matter
3. **Broadened to all AI agents** (March 14 reframing) → right direction, but still too broad
4. **Sharpened to autonomous agents** (this pivot) → the real question

Traditional algorithms are deterministic. You can predict what a vol-targeting model will do because it follows a formula. Autonomous agents are different: they reason about market state, infer other agents' behavior, and adapt. When 50 LLM agents are trading the same market, do they converge on the same strategy without being told to? Do they create market regimes that wouldn't exist without them? Do they learn to exploit each other?

These questions can't be answered by looking at historical market data from the pre-agent era. They require a simulation framework where you can control agent behavior and test counterfactuals.

### What Changed

- **Old Q1 deprecated.** The vol-shock persistence/reversal research becomes a footnote — the investigation that revealed where to look. Code stays in `harbor/abf/q1/` as historical work.
- **Simulation moved to center stage.** The agent simulation framework went from late-stage addition (old H5) to immediate priority (new H3). It's the primary research instrument, not a supporting tool.
- **LLM agents moved from stub to focus.** In the old plan, LLM agents were "Phase 3 scope" with a `NotImplementedError`. Now they're the centerpiece of H4.
- **HARBOR became a participant.** HARBOR-as-agent wraps the existing portfolio logic and competes against autonomous agents. This answers the natural question: "does the system you built actually work when autonomous agents are in the market?"
- **New three-question research arc:**
  - Q1: **Emergent Coordination** — do independent agents converge?
  - Q2: **Regime Manufacturing** — do agent populations create regimes?
  - Q3: **Adversarial Adaptation** — do agents learn to exploit each other?

### The Design Decision

I chose an evolutionary pivot — layer the new direction on top of validated infrastructure rather than starting over. H1 and H2 remain untouched. The 205 passing tests still pass. The old Q1 code stays in the repo. This is the same principle as before: don't break what works.

But the research direction is fundamentally different. The old plan studied whether algorithms create patterns in markets. The new plan studies whether autonomous agents — systems that reason, adapt, and interact — create emergent dynamics that algorithms never could.

### What I Expect to Learn

Three possible outcomes, all interesting:

1. **Convergence without coordination.** Independent autonomous agents naturally converge on similar strategies because they see the same data and reason similarly. This would mean autonomous agent proliferation inherently creates systemic risk — not from explicit coordination, but from shared cognition.

2. **Manufactured regimes.** Agent populations create momentum, mean-reversion, and volatility clustering that wouldn't exist without them. Markets with autonomous agents have fundamentally different statistical properties than markets with only traditional algorithms.

3. **Adversarial equilibrium or instability.** When agents adapt to each other, the market either stabilizes (agents find complementary niches) or destabilizes (arms race). The retail investor impact depends on which.

The honest answer: I don't know which of these will emerge. The simulation framework is designed to find out.

---

---

## Chapter 6: The Build (April 2026)

### Why the Restructure

After the pivot to autonomous agents in Chapter 5, the old phase plan (H3 → H4 → H5 → A3 → A4...) stopped making sense. The problem: it was incremental in the wrong way. It deferred the experiment infrastructure to late stages (old H5) and built toward it through a sequence of isolated pieces. But running real agent experiments requires the full experiment loop to exist first — you need YAML configs, a runner, recording, metrics, and a results store before you can do anything reproducible. Building one component at a time would mean weeks of scaffolding before any experiment could run.

The decision: replace the phase roadmap entirely with a five-layer architecture, build the experiment infrastructure (Layer 4) in one pass, and make it functional before running any experiments. This is the "homelab" — a personal research cluster for empirical multi-agent finance.

A second motivation: the existing `harbor/agents/` code was not yet wired to a reproducible experiment loop. The market environment, rule agents, and metrics all existed, but they were called from ad-hoc scripts. The restructure gave them a proper home inside a normalized, config-driven experiment system.

### What Was Built

**`harbor/agents/` — completed:**
- `MarketEnvironment` with price-impact model (linear temporary + square-root permanent), multi-asset state, order matching
- `BaseAgent` abstract interface: `observe() → decide() → act()`
- `MomentumAgent`, `MeanReversionAgent`, `VolTargetAgent` — rule-based baselines
- `PopulationMetrics` — crowding index, flow imbalance, regime labels
- `Simulation` — multi-agent simulation runner

**`harbor/homelab/` — built from scratch:**

*Venue layer:* `VenueSnapshot` (normalized state schema), `EquityVenue` (adapter wrapping `MarketEnvironment`). The venue produces a clean, consistent observation for any consumer — agent or runner — regardless of what's underneath.

*Agent layer:* Four composable protocols (`Observable`, `Configurable`, `ToolUser`, `BudgetAware`). `LegacyAgentAdapter` bridges the existing `BaseAgent` subclasses to the new `Observable` protocol without touching their implementation. `AgentRegistry` constructs agents from YAML config entries.

*Experiment config:* `ExperimentConfig` dataclass with full serialization. `from_yaml()` loads a YAML file; `to_dict()` serializes it for recording and replay.

*Runner:* `ExperimentRunner` — the core loop. It derives child seeds deterministically via `np.random.SeedSequence`, builds venue and agents, runs the simulation, computes metrics, and finalizes recording. Returns an `ExperimentResult` with prices, returns, per-agent weights, orders, and metrics.

*Batch and ablation:* `BatchRunner` runs a list of configs sequentially. `AblationRunner` generates the full Cartesian product of a parameter grid (specified via dot-path notation) and runs each variant.

*Recording:* `Recorder` protocol with `NoopRecorder` (default, silent) and `JsonlRecorder` (writes one record per step). The backend is pluggable — this design anticipates Flight integration for trace replay and observability.

*Evaluation:* `MetricsRegistry` with named metric functions and `compute_all()`. `ExperimentSummary` aggregates result sets.

*Results store:* `ResultsStore` — persists results with metadata.

*CLI:* `python -m harbor.homelab experiment.yaml` runs any experiment from a YAML config.

### Key Design Decisions

**Homelab orchestrates, doesn't absorb.** `harbor/agents/`, `harbor/risk/`, and `harbor/portfolio/` keep their module paths. The homelab wraps them via adapters and protocols rather than pulling their code in. This means the existing 205 tests kept passing during the build — no rewrite-driven breakage.

**Protocol composition over inheritance.** Agents compose only the protocols they implement. An agent that doesn't use tools doesn't implement `ToolUser`. This makes the agent API extensible without forcing every agent type to satisfy a large abstract base class. LLM agents (future) will add `ToolUser` and `BudgetAware` without changing anything else.

**`LegacyAgentAdapter` as the bridge.** Rather than rewriting the existing rule agents to satisfy new protocols, a thin adapter converts `VenueSnapshot → MarketState` and delegates to the inner `BaseAgent`. This is the software engineering equivalent of "don't break what works."

**Pluggable recording backend.** `JsonlRecorder` writes per-step traces. The `Recorder` protocol is intentionally minimal so that a Flight-backed recorder can slot in without changing the runner. Trace observability is a first-class concern — not a later add-on.

**`SeedSequence` for determinism.** A single master seed in the YAML config spawns child seeds for the venue and each agent via `np.random.SeedSequence`. Every experiment is reproducible from one integer.

### What Was Learned

Building the experiment infrastructure before running experiments is the right order. Every research decision — what to measure, how to compare, how to reproduce — is cleaner when the scaffolding already exists. The temptation is to start running simulations immediately and wire up infrastructure later. That path leads to ad-hoc scripts that can't be reproduced or compared.

The `LegacyAgentAdapter` pattern was worth the extra indirection. It let the new system inherit all the work done on `harbor/agents/` without a rewrite, and it's reusable: any future "bring your own agent" implementation follows the same pattern.

Protocol-based composition is significantly cleaner than a deep inheritance hierarchy for agents. The existing `BaseAgent` class had a fixed interface. The new protocol approach lets different agent types (heuristic, LLM, RL) share only the interfaces they actually need.

---

## Technical Timeline

| Date | Version | Milestone |
|------|---------|-----------|
| Oct 2025 | — | Project started for Wharton Global Investment Competition |
| Feb 1, 2026 | v0.1.0-dev | Core quant stack: data, risk, portfolio, backtest |
| Feb 13, 2026 | — | ABF PRD v1 written |
| Feb 24, 2026 | — | Phase 2 sprint plan created |
| Feb 27, 2026 | v0.2.0-dev | Q1 pipeline, advanced risk engine, ML scaffolding, 187 tests |
| Mar 12, 2026 | — | Massive API integration design (data infrastructure upgrade) |
| Mar 14, 2026 | — | First reframing: broadened to all AI agents, added retail focus |
| Mar 15, 2026 | v0.3.0-dev | Classical baselines (GARCH/EWMA), multicollinearity fix, 205 tests |
| Mar 16, 2026 | — | **Pivot to autonomous agents: new Q1/Q2/Q3, simulation-first research** |
| Apr 1, 2026 | — | **5-layer restructure: `harbor/agents/` complete, `harbor/homelab/` built** |
| Apr 1, 2026 | — | 278 tests, CLI working: `python -m harbor.homelab experiment.yaml` |

## Architecture at a Glance

```
harbor/
  data/              Universe, price loaders, caching
  risk/              Covariance, HRP, Monte Carlo, regime detection, scenarios, decomposition
  portfolio/         Optimization, constraints, allocation
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters + baselines, behavioral RL agents
  agents/            Agent simulation: environment, rule agents, metrics (Layer 2 core)
  homelab/           Experiment infrastructure (Layer 4)
    venue/             Normalized venue abstraction (EquityVenue)
    agent/             Protocol-based agent API + LegacyAgentAdapter
    recording/         Pluggable trace recording (noop, JSONL)
    evaluation/        Metrics registry and experiment summaries
    results/           Results store
    config.py          YAML experiment config loader
    batch.py / ablation.py  Batch and ablation runners
    runner.py          ExperimentRunner
    __main__.py        CLI entry point
  abf/               ABF research utilities (Q1 deprecated)
```

## Metrics

- **Tests:** 278 passing (as of April 2026)
- **Modules:** 9 top-level packages
- **Research questions:** 3 (emergent coordination, regime manufacturing, adversarial adaptation)
- **Architecture restructures:** 1 (phase roadmap → 5-layer platform, April 2026)
- **Scope pivots:** 2 (broadening to all AI agents → sharpening to autonomous agents)
