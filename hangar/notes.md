# `harbor/` Package Notes

## Explanation
`harbor/` is the core Python package. It is intentionally split by responsibility so each stage of the quant workflow has a narrow interface.

Conceptually the package behaves like a directed graph:
- `harbor.data` -> `harbor.risk` -> `harbor.portfolio` -> `harbor.backtest`
- `harbor.abf` and `harbor.features` sit across this path for research questions.
- `harbor.ml` is an extension path for later phases (not fully implemented yet).

Current implemented Phase 1 core modules:
- `data`: production of clean input time series.
- `risk`: covariance, HRP, shock proxy, Monte Carlo VaR/CVaR.
- `portfolio`: MVO, risk parity, HRP weight interfaces.
- `backtest`: execution simulation with transaction costs + metrics.

This decomposition reduces coupling:
- risk module does not need to know about transaction costs.
- backtest module does not need to know where data came from.
- portfolio module can switch risk engines without changing backtest logic.

## Data/Control Flow
1. Load universe and prices.
2. Compute returns.
3. Estimate risk objects (`Sigma`, cluster structure, proxies).
4. Produce target weights.
5. Simulate trading and evaluate output metrics.

## Interfaces To Learn First
- `harbor.data.load_sp500_prices`
- `harbor.risk.estimate_covariance`
- `harbor.portfolio.hrp_weights`
- `harbor.backtest.run_cross_sectional_backtest`

## Your Notes
- 
