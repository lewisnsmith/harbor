# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-dev] — 2026-02-27

### Added
- **ABF Q1 pipeline** (`harbor.abf.q1`): shock detection, local projections (Newey-West HAC), cumulative abnormal return computation, robustness sweep, and figure generation.
- **ML scaffolding** (`harbor.ml`): LSTM/GRU volatility forecasters (`harbor.ml.volatility`), deep RL behavioral agents with behavioral reward shaping (`harbor.ml.behavior_agents`), and file-based checkpoint registry (`harbor.ml.checkpoints`).
- **Cross-sectional backtest engine** (`harbor.backtest.engine`): rolling-window backtest loop with configurable lookback, rebalance frequency, and transaction costs.
- **Cumulative abnormal return** metric (`harbor.backtest.metrics.cumulative_abnormal_return`): standard event-study CAR computation for ABF Q1.
- **CRSP/WRDS loader stub** (`harbor.data.load_prices.load_crsp_prices`): interface placeholder for institutional data access.
- **Makefile** with `install`, `test`, `lint`, `q1`, `h1`, and `all` targets for reproducibility.
- **CI coverage reporting** via `pytest-cov` with terminal and XML output.
- ABF Q1 experiment orchestrator (`experiments/abf_q1_main.py`).
- H1 end-to-end HRP backtest script (`experiments/h1_end_to_end_hrp_backtest.py`).
- Comprehensive test suite for data, risk, portfolio, backtest, ML volatility, ML agents, and ABF Q1 modules.
- Config scaffolding for ABF shock and regime definitions (`configs/abf/`).

### Changed
- Restructured documentation: consolidated `iteration-history.md` into `journal.md`, updated `docs/README.md` index.
- Updated `docs/plan.md` to reflect actual implementation state: ML modules labeled as experimental scaffolding.
- Pinned dependency versions in `requirements.txt` for reproducibility.
- Updated README with research contribution section, data limitations, reproducibility instructions, and CI badge.
- Cleaned `pyproject.toml`: removed references to deprecated `harbor.signals` and `harbor.research` wrappers.
- Improved `__all__` exports across all public modules.

### Removed
- `docs/iteration-history.md` (merged into `journal.md`).
- Deprecated `harbor.signals` and `harbor.research` wrapper modules.
- Development artifacts: `proxy.mjs`, `start_jupyter.py`, `notebooks/Untitled.ipynb`, `v1_alpaca_performance.png`.
- `alpaca-trade-api` from `requirements.txt` (v1 paper trading dependency, not used in current phase).

## [0.1.0-dev] — 2026-02-01

### Added
- Initial project structure with `harbor` package.
- `harbor.data`: S&P 500 universe loaders with survivorship-bias-aware fallbacks, chunked concurrent price fetching via `PriceLoader`, risk-free rate proxy, local Parquet/pickle cache.
- `harbor.risk`: sample and Ledoit-Wolf shrinkage covariance estimators, HRP allocation, Monte Carlo VaR/CVaR simulation, regime detection (vol shocks, correlation spikes).
- `harbor.portfolio`: mean-variance, risk parity, and HRP weight interfaces with regime-aware position sizing.
- `harbor.backtest`: Sharpe, Sortino, Calmar, max drawdown, and win rate metrics.
- ABF PRD (`docs/abf-prd.md`) and project roadmap (`docs/plan.md`).
- CI pipeline with ruff linting, compileall, and pytest.
