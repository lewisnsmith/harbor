# HARBOR — Hierarchical Adaptive Risk-Based Optimization Routine

> A personal, research-driven asset management framework built from scratch — from high-school investment competitions to a semi-autonomous portfolio system targeting retail investors.

**Author:** Lewis Smith

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
- **ML Extensions** — Neural volatility forecasting (LSTM/GRU) and deep RL behavioral agents (in development)

---

## Research Track: Artificial Behavioral Finance (ABF)

Beyond portfolio management, HARBOR serves as the empirical platform for an original research agenda: **do ML-driven trading strategies manufacture market dynamics that traditional behavioral finance can't explain?**

Core research questions (full specification in [`docs/abf-prd.md`](docs/abf-prd.md)):

1. **Manufactured autocorrelation** — Do volatility-targeting and trend-following models create momentum regimes that subsequently reverse?
2. **Synchronized crowding** — Does signal similarity across systematic agents amplify drawdowns and correlation spikes?
3. **Alpha decay** — Does widespread model deployment accelerate factor decay? *(stretch goal)*

The ABF track is designed to produce working-paper-quality output with reproducible code pipelines, targeting faculty review and potential collaboration at UCLA Anderson and IPAM.

---

## Repository Structure

```
harbor/            Core Python package
  data/              Universe, price loaders, caching
  risk/              Covariance, HRP, Monte Carlo, regime detection
  portfolio/         Optimization and allocation interfaces
  backtest/          Engine, metrics, experiment runners
  ml/                Volatility forecasters, behavioral RL agents
  abf/               ABF research experiment utilities
notebooks/         Research and experimentation notebooks
experiments/       End-to-end scripts and prototypes
configs/           Config-driven experiment definitions (ABF shock/regime specs)
dashboard/         Portfolio monitoring dashboard
docs/              Project documentation
  journal.md         Development history and lessons learned
  abf-prd.md         ABF research product requirements document
  plan.md            Roadmap and phase tracking
  iteration-history.md   Raw iteration-by-iteration technical log
data/              Universe membership and cached data
tests/             Unit and integration test suite
research/          Research notes and references
```

---

## Current Status

| Track | Phase | Status |
|-------|-------|--------|
| **HARBOR** | H1 — Core Quant Stack | Complete — data, risk, portfolio, backtest all implemented |
| **ABF** | A1/A2 — Spec + early Q1 | Baseline configs and shock utilities scaffolded |

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

**Launch notebooks:**

```bash
jupyter lab
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`journal.md`](docs/journal.md) | How the project evolved — personal narrative and lessons learned |
| [`abf-prd.md`](docs/abf-prd.md) | ABF research specification, hypotheses, test plans, and UCLA outreach |
| [`plan.md`](docs/plan.md) | Technical roadmap with phase checklists and milestone tracking |
| [`iteration-history.md`](docs/iteration-history.md) | Raw technical log of each iteration's capabilities and faults |
| [`notes.md`](notes.md) | Phase 1 component learning notes |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned release history |

---

## Disclaimer

This project is for **research and educational purposes only** and does not constitute financial advice. HARBOR is a personal learning project documenting one person's journey through quantitative finance, portfolio theory, and machine learning — not a production trading system.
