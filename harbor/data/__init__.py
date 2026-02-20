"""
harbor.data — Price data loading, S&P 500 universe management, and feature storage.

Planned functionality (Phase H1):
- Load adjusted daily close prices for S&P 500 constituents via yfinance.
- Manage point-in-time universe snapshots to avoid look-ahead bias.
- Cache raw and processed data to disk to limit API calls.

ABF extensions (Phase A2+):
- Expose shock-date series consumed by harbor.abf.q1.
- Provide rolling constituent windows for crowding analysis (ABF Q2).
"""

from harbor.data.load_prices import (
    load_sp500_prices,
    load_sp500_tickers,
    load_risk_free_rate,
)

__all__ = [
    "load_sp500_prices",
    "load_sp500_tickers",
    "load_risk_free_rate",
]
