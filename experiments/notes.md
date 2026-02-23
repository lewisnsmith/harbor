# `experiments/` Notes

## Explanation
`experiments/` contains runnable scripts that validate milestones or test targeted scenarios outside notebooks.

Current key script:
- `h1_end_to_end_hrp_backtest.py`

How the H1 script functions:
1. Load point-in-time universe.
2. Load historical prices.
3. Convert prices to returns.
4. Build rolling HRP weights (covariance from lookback window).
5. Run transaction-cost-aware cross-sectional backtest.
6. Compute and print performance metrics.
7. Compute Monte Carlo VaR/CVaR for latest portfolio.

Mathematical pieces used:
- return transform: `P_t/P_{t-1} - 1`
- covariance shrinkage estimate
- HRP allocation mechanics
- backtest net return with cost drag
- VaR/CVaR tail-risk estimation

Why scripts here matter:
- They are the quickest reproducible path to verify phase exit criteria.

## Your Notes
- 
