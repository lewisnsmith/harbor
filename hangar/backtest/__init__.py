"""
harbor.backtest — Backtest engine, metrics, and experiment runners.

Phase H1 deliverables:
- Cross-sectional backtest loop with transaction costs.
- Standard performance and risk metrics (Sharpe, max drawdown, etc.).

Phase H2+ extensions:
- Regime-aware vs baseline mode toggling (ABF integration).
- Config-driven scenario runs.
"""

from harbor.backtest.engine import BacktestResult, run_cross_sectional_backtest
from harbor.backtest.metrics import (
    calmar_ratio,
    cumulative_abnormal_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

__all__ = [
    "BacktestResult",
    "calmar_ratio",
    "cumulative_abnormal_return",
    "max_drawdown",
    "run_cross_sectional_backtest",
    "sharpe_ratio",
    "sortino_ratio",
    "win_rate",
]
