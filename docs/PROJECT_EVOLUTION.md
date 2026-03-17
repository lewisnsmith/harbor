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

3. **The retail investor angle:** As AI agents proliferate, the people most affected are retail investors who lack the tools to detect or respond to AI-driven market dynamics. This gave the project a purpose beyond academic curiosity.

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

## Architecture at a Glance

```
harbor/
  data/              Universe, price loaders, caching, Massive API client
  risk/              Covariance, HRP, Monte Carlo, regime detection, scenarios, decomposition
  portfolio/         Optimization, constraints, allocation
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters + baselines, behavioral RL agents
  abf/               ABF research: old Q1 (deprecated), new Q1/Q2/Q3 (planned)
  agents/            Agent simulation framework: environment, autonomous agents, experiments
  retail/            Retail impact metrics and analysis (planned)
```

## Metrics

- **Tests:** 205 passing (as of March 2026)
- **Modules:** 8 top-level packages
- **Research questions:** 3 (emergent coordination, regime manufacturing, adversarial adaptation)
- **Phases complete:** 4 of 10 (H1, H2, A1, A2)
- **Scope pivots:** 2 (broadening to all AI agents → sharpening to autonomous agents)
