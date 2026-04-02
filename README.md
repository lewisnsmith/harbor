# HARBOR

[![CI](https://github.com/lewis-smith/harbor/actions/workflows/ci.yml/badge.svg)](https://github.com/lewis-smith/harbor/actions/workflows/ci.yml)

A research platform for studying how autonomous trading agents reshape market dynamics.

**Author:** Lewis Smith

---

## Research Contribution

HARBOR serves as the empirical platform for an original research agenda in **Artificial Behavioral Finance (ABF)**: investigating whether autonomous trading agents — LLM-powered, RL-based, and tool-using — reshape market dynamics in ways traditional behavioral finance cannot explain. Unlike conventional algorithmic trading, these agents reason, adapt, and interact strategically, creating emergent coordination, manufactured regimes, and adversarial dynamics. The framework combines risk modeling (HRP, Monte Carlo VaR/CVaR, regime detection) with a multi-agent market simulation for causal testing. See the [full research specification](docs/abf-prd.md) and [simulation results](results/agent_simulation/) for details.

---

## The Story

HARBOR didn't start as a grand plan. It started as a Monte Carlo simulation I pieced together for the Wharton Global High School Investment Competition because I wanted something more rigorous than gut-feel stock picks for my report. That first script was naive, but it planted a seed. I realized this quantitative rigor was something necessary for the money my grandma had given me to invest for her. I wanted to build it right — rigorous enough to trust, simple enough to verify. 

Over eight iterations, the project grew — sometimes deliberately, sometimes because I encountered a fascinating piece of quantitative theory and couldn't resist implementing it. Factor screening inspired by AQR. Mean-variance optimization. Black-Litterman. Hierarchical Risk Parity. Hidden Markov Models for regime detection. Reinforcement learning for adaptive weighting. Each layer addressed a failure I observed in the previous version. Each layer sounded so intriguing I couldn't resist implementing it.

Eventually, the system had become theoretically impressive but practically fragile. The turning point was realizing that sophistication I couldn't fully verify was worse than simplicity I could trust. The current version strips back to the components with proven out-of-sample value and prioritizes robustness, interpretability, and reliability — qualities that matter regardless of context. That discipline — knowing when to cut — is what made the deeper research questions visible.

The full development history is documented in [`docs/journal.md`](docs/journal.md).

---

## What HARBOR Does

HARBOR is a modular Python framework for **risk-first portfolio construction and research**. Its modules are the infrastructure layer of the ABF research platform:

- **Data** — S&P 500 universe with survivorship-bias-aware loaders and local caching
- **Risk** — Covariance estimation (sample + shrinkage), Hierarchical Risk Parity, Monte Carlo VaR/CVaR, regime detection
- **Portfolio** — Mean-variance, risk parity, and HRP allocation with configurable constraints
- **Backtest** — Cross-sectional backtesting engine with transaction costs and standard performance/risk metrics
- **ML Extensions** — Neural volatility forecasting (LSTM/GRU) and deep RL behavioral agents
- **Agent Simulation** — Multi-agent market environment where autonomous agents interact, generating synthetic price impact and crowding dynamics for causal testing
- **Homelab** — Reproducible experiment infrastructure: YAML configs, ExperimentRunner, batch/ablation runners, JSONL trace recording, metrics registry, results store. Run any experiment with `python -m harbor.homelab experiment.yaml`

---

## Observability: Flight

Agent traces recorded by Harbor's `JsonlRecorder` are designed for consumption by [Flight](link-tbd) — a purpose-built trace capture tool. Flight enables step-level replay, trace inspection, and debugging outside the experiment loop. The `Recorder` protocol is intentionally minimal so Flight can be swapped in without changing the runner.

---

## Artificial Behavioral Finance (ABF)

The central question: **do autonomous trading agents reshape market dynamics in ways traditional behavioral finance can't explain?**

Core research questions (full specification in [`docs/abf-prd.md`](docs/abf-prd.md)):

1. **Emergent Coordination** — Do independent autonomous agents converge on similar strategies, creating crowding and herding without explicit communication?
2. **Regime Manufacturing** — Do agent populations CREATE market regimes (volatility clusters, momentum/reversal cycles) rather than merely responding to them?
3. **Adversarial Adaptation** — Do agents learn to exploit each other's strategies, generating predator-prey dynamics and arms-race escalation?

The ABF track uses a simulation-first methodology: synthetic multi-agent markets with controlled populations enable causal identification of agent-driven effects, producing research-quality output with the aim of faculty review and potential collaboration.

**The retail investor angle is two-sided.** On one hand, autonomous agents operating in markets may create dynamics — crowding, manufactured regimes, adversarial arms races — that retail investors can't detect or respond to, concentrating risk on those least equipped to handle it. On the other hand, autonomous agents *as tools* may expand retail access to strategies previously gated behind institutional infrastructure: regime-aware allocation, dynamic risk sizing, and systematic rebalancing that individual investors couldn't implement manually. Whether this democratization offsets the structural disadvantage is an open empirical question — and one HARBOR is positioned to test, as HARBOR-as-agent wraps its own portfolio logic and competes directly against autonomous agents in simulation.

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
  homelab/           Reproducible experiment infrastructure
    venue/             Normalized venue abstraction (EquityVenue)
    agent/             Protocol-based agent API + LegacyAgentAdapter
    recording/         Pluggable trace recording (noop, JSONL)
    evaluation/        Metrics registry and experiment summaries
    results/           Results store
    config.py          YAML experiment config loader
    batch.py           Batch experiment runner
    ablation.py        Ablation / parameter-sweep runner
    runner.py          ExperimentRunner: YAML → run → traces + metrics
    __main__.py        CLI entry point
  abf/               ABF research experiment utilities (legacy Q1 pipeline)
notebooks/         Research and experimentation notebooks
experiments/       End-to-end scripts and prototypes
configs/           YAML experiment configs
dashboard/         Portfolio monitoring dashboard
results/           Committed research outputs (figures, tables)
docs/              Project documentation
  journal.md         Development history and lessons learned
  abf-prd.md         ABF research product requirements document
  plan.md            Architecture and completed-work reference
data/              Universe membership and cached data
tests/             Unit and integration test suite
research/          Research notes and references
```

---

## Current Status

| Layer | Module | Status |
|-------|--------|--------|
| Core Quant Stack | `harbor.data`, `harbor.risk`, `harbor.portfolio`, `harbor.backtest` | Complete |
| Advanced Risk | `harbor.risk` — regime detection, scenarios, decomposition | Complete |
| Agent Simulation | `harbor.agents` — environment, rule agents, metrics | Complete |
| Experiment Infrastructure | `harbor.homelab` — YAML runner, batch, ablation, recording, metrics | Complete |
| ML Extensions | `harbor.ml` — vol forecasters, RL agents | Experimental scaffolding |
| ABF Q1 Pipeline | `harbor.abf` | Legacy/deprecated |

278 tests passing. See [`docs/plan.md`](docs/plan.md) for architecture details.

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

**Run an experiment via the homelab CLI:**

```bash
python -m harbor.homelab configs/your_experiment.yaml
```

**Run the H1 end-to-end baseline:**

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
| [`plan.md`](docs/plan.md) | Architecture reference: 5-layer design, completed modules, current state |
| [`CHANGELOG.md`](CHANGELOG.md) | Versioned release history |

---

## Disclaimer

This project is for **research and educational purposes only** and does not constitute financial advice. HARBOR is a personal learning project documenting one person's journey through quantitative finance, portfolio theory, and machine learning — not a production trading system.
