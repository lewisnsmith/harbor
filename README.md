# HARBOR — Hierarchical Agentic Risk Based Optimization Routine

[![CI](https://github.com/lewis-smith/harbor/actions/workflows/ci.yml/badge.svg)](https://github.com/lewis-smith/harbor/actions/workflows/ci.yml)

> A personal, research-driven asset management framework built from scratch — from high-school investment competitions to a semi-autonomous portfolio system targeting retail investors.

**Author:** Lewis Smith

---

## Research Contribution

HARBOR serves as the empirical platform for an original research agenda in **Artificial Behavioral Finance (ABF)**: investigating whether autonomous trading agents — LLM-powered, RL-based, and tool-using — reshape market dynamics in ways traditional behavioral finance cannot explain. Unlike conventional algorithmic trading, these agents reason, adapt, and interact strategically, creating emergent coordination, manufactured regimes, and adversarial dynamics. The framework combines institutional-grade risk modeling (HRP, Monte Carlo VaR/CVaR, regime detection) with a multi-agent market simulation for causal testing. See the [full research specification](docs/abf-prd.md) and [simulation results](results/agent_simulation/) for details.

---

## The Story

HARBOR didn't start as a grand plan. It started as a Monte Carlo simulation I pieced together for the Wharton Global High School Investment Competition because I wanted something more rigorous than gut-feel stock picks for my report. That first script was naive, but it planted a seed. I realized this quantitative rigor was something necessary for the money my grandma had gave me to invest for her. As the system evolved, I sought to make it institutional quality, but tailored to smaller portfolios and with the transparency and customizability that a retail investor can actually depend on. 

Over eight iterations, the project grew — sometimes deliberately, sometimes because I encountered a fascinating piece of quantitative theory and couldn't resist implementing it. Factor screening inspired by AQR. Mean-variance optimization. Black-Litterman. Hierarchical Risk Parity. Hidden Markov Models for regime detection. Reinforcement learning for adaptive weighting. Each layer addressed a failure I observed in the previous version. Each layer sounded so intriguing I couldn't resist implementing it.

Eventually, the system had become theoretically impressive but practically fragile. The turning point was realizing that sophistication I couldn't fully verify was worse than simplicity I could trust. The current version strips back to the components with proven out-of-sample value and prioritizes robustness, interpretability, and reliability — qualities that matter when the goal is a system a retail investor can actually depend on.

The full development history is documented in [`docs/journal.md`](docs/journal.md).

---

## What HARBOR Does

HARBOR is a modular Python framework for **risk-first portfolio construction and management**, designed for individual/retail-level investors:

- **Data** — S&P 500 universe with survivorship-bias-aware loaders and local caching
- **Risk** — Covariance estimation (sample + shrinkage), Hierarchical Risk Parity, Monte Carlo VaR/CVaR, regime detection
- **Portfolio** — Mean-variance, risk parity, and HRP allocation with configurable constraints
- **Backtest** — Cross-sectional backtesting engine with transaction costs and standard performance/risk metrics
- **ML Extensions** — Neural volatility forecasting (LSTM/GRU) and deep RL behavioral agents
- **Agent Simulation** — Multi-agent market environment where autonomous agents interact, generating synthetic price impact and crowding dynamics for causal testing

---

## Research Track: Artificial Behavioral Finance (ABF)

Beyond portfolio management, HARBOR serves as the empirical platform for an original research agenda: **do autonomous trading agents reshape market dynamics in ways traditional behavioral finance can't explain?**

Core research questions (full specification in [`docs/abf-prd.md`](docs/abf-prd.md)):

1. **Emergent Coordination** — Do independent autonomous agents converge on similar strategies, creating crowding and herding without explicit communication?
2. **Regime Manufacturing** — Do agent populations CREATE market regimes (volatility clusters, momentum/reversal cycles) rather than merely responding to them?
3. **Adversarial Adaptation** — Do agents learn to exploit each other's strategies, generating predator-prey dynamics and arms-race escalation?

The ABF track uses a simulation-first methodology: synthetic multi-agent markets with controlled populations enable causal identification of agent-driven effects, producing working-paper-quality output targeting faculty review and potential collaboration.

---

## Repository Structure

```
harbor/            Core Python package
  data/              Universe, price loaders, caching
  risk/              Covariance, HRP, Monte Carlo, regime detection
  portfolio/         Optimization and allocation interfaces
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters, behavioral RL agents
  agents/            Multi-agent simulation: environment, agents, metrics
  abf/               ABF research experiment utilities (legacy Q1 pipeline)
notebooks/         Research and experimentation notebooks
experiments/       End-to-end scripts and prototypes
configs/           Config-driven experiment definitions (ABF shock/regime specs)
dashboard/         Portfolio monitoring dashboard
results/           Committed research outputs (figures, tables)
docs/              Project documentation
  journal.md         Development history and lessons learned
  abf-prd.md         ABF research product requirements document
  plan.md            Roadmap and phase tracking
data/              Universe membership and cached data
tests/             Unit and integration test suite
research/          Research notes and references
```

---

## Current Status

| Track | Phase | Status |
|-------|-------|--------|
| **HARBOR** | H1 — Core Quant Stack | Complete — data, risk, portfolio, backtest implemented and tested |
| **HARBOR** | H2 — Advanced Risk & Simulation | Complete — regime detection, scenarios, decomposition |
| **HARBOR** | H3 — Agent Simulation Core | Complete — market environment, rule-based agents, metrics, demo figure |
| **ABF** | A1/A2 — Q1 Pipeline | Legacy baseline — preliminary results committed |
| **ML** | Experimental | Scaffolding implemented (vol forecasting, RL agents); validation pending |

See [`docs/plan.md`](docs/plan.md) for the full roadmap and milestone table.

---

## Quickstart

**Prerequisites:** Python 3.9+ (3.11 recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

**Run the Phase H1 end-to-end baseline:**

```bash
python3 experiments/h1_end_to_end_hrp_backtest.py --start 2020-01-01 --max-assets 50
```

**Run the test suite:**

```bash
pytest -q
```

### Reproducibility

A `Makefile` is provided for one-command reproducibility:

```bash
make install    # Create venv and install dependencies
make test       # Run pytest
make lint       # Run ruff
make q1         # Run ABF Q1 pipeline end-to-end
make h1         # Run H1 HRP backtest pipeline
make h3         # Run H3 agent simulation demo → results/agent_simulation/
make all        # install + lint + test + run pipelines
```

---

## Data Sources & Limitations

- **Development proxy:** V1 uses [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance) for price data. This is adequate for pipeline development and method validation but not for publication-quality inference.
- **Survivorship bias:** The default universe loader scrapes current S&P 500 constituents from Wikipedia when historical membership data is unavailable. This introduces survivorship bias; a warning is emitted at runtime.
- **Production target:** I hope to get CRSP/WRDS for survivorship-bias-free historical constituents. The `load_crsp_prices()` stub in `harbor.data` defines the interface; integration requires institutional access.
- **Risk-free rate:** Proxied via 13-week T-bill yield (`^IRX`), converted and forward-filled to daily frequency.

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`journal.md`](docs/journal.md) | How the project evolved — personal narrative and lessons learned |
| [`abf-prd.md`](docs/abf-prd.md) | ABF research specification, hypotheses, test plans, and UCLA outreach |
| [`plan.md`](docs/plan.md) | Technical roadmap with phase checklists and milestone tracking |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned release history |

---

## Disclaimer

This project is for **research and educational purposes only** and does not constitute financial advice. HARBOR is a personal learning project documenting one person's journey through quantitative finance, portfolio theory, and machine learning — not a production trading system.
