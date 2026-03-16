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

## Chapter 4: The Scientist (March 2026 – Present)

### The Plan

Phase 2 uses a two-track parallel structure:

**Track 1 (Empirical — A3):** Test Q2 (crowding and correlation instability) using real market data. Build crowding proxies, detect correlation spikes, run predictive regressions and Granger causality tests. This track depends only on the completed H2 risk engine.

**Track 2 (Simulation — H3 → H4 → H5):** Validate the ML models (H3), validate behavioral agents produce distinguishable behavior (H4), then build a full agent simulation framework (H5) where heterogeneous agent populations interact in a market environment.

**Convergence (A4):** The payoff. Run the same analysis on simulated data that was run on real data. Do agent populations, with no access to real market data, independently produce the same statistical signatures found empirically? If yes, that's causal evidence. If no, that's also a publishable result.

The two-track structure was chosen because:
- A3 doesn't depend on ML validation — it can start immediately
- The convergence point (A4) is the strongest possible result: "simulated agents reproduce real market patterns"
- Parallel work reduces wall-clock time without sacrificing rigor

### What's Being Built

- **Agent simulation framework** with a price-impact market model, abstract agent interface, rule-based and ML-based agent types, and population management
- **Causal experiments** that are impossible with real markets: ablation (remove agent types and observe effect), dose-response (vary agent concentration), heterogeneity testing (diverse vs homogeneous populations)
- **Pattern matching** between simulated and real data using the same metrics and statistical tests

### Agentic Trading Testing Plan

The H5 agent simulation framework enables systematic testing of how different AI agent configurations affect market dynamics:

1. **Homogeneous populations:** What happens when many agents use the same strategy? Does crowding emerge mechanically?
2. **Mixed populations:** How do different agent types interact? Do momentum agents create opportunities for mean-reversion agents, or does everyone lose?
3. **Stress scenarios:** When external shocks hit, do agent interactions amplify or dampen the response?
4. **Convergence dynamics:** If agents learn and adapt, do they converge on similar strategies over time?
5. **LLM agent integration (future):** When LLM-based agents enter the simulation, do they behave like traditional systematic agents or create novel dynamics?

### What I Expect to Learn

The honest answer: I don't know yet. The empirical evidence from A2 suggests that vol-targeting behavior manufactures autocorrelation, but whether the simulation will reproduce this is an open question. A null result — simulated agents don't produce real-world patterns — would mean the real-world patterns are driven by something deeper than agent behavior alone. That would redirect the research toward fundamental market structure, which is equally interesting.

This uncertainty is the point. The project isn't built to confirm a hypothesis — it's built to test one.

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
| Mar 14, 2026 | — | Project reframing: broadened to all AI agents, added retail focus |
| Mar 15, 2026 | — | Phase 2 design: two-track parallel (empirical + simulation) |

## Architecture at a Glance

```
harbor/
  data/              Universe, price loaders, caching, Massive API client
  risk/              Covariance, HRP, Monte Carlo, regime detection, scenarios, decomposition
  portfolio/         Optimization, constraints, allocation
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters (experimental), behavioral RL agents (experimental)
  abf/               ABF research: Q1 (complete), Q2 (in progress), agent validation (planned)
  agents/            Agent simulation framework (planned)
  retail/            Retail impact metrics and analysis (planned)
```

## Metrics

- **Tests:** 187 passing (as of v0.2.0-dev), targeting ~290 by end of Phase 2
- **Modules:** 8 top-level packages
- **Research questions:** 3 (Q1 answered, Q2 in progress, Q3 deferred)
- **Phases complete:** 4 of 11 (H1, H2, A1, A2)
