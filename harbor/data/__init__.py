"""
harbor.data — Price data loading, S&P 500 universe management, and feature storage.

Planned functionality (Phase H1):
- Load adjusted daily close prices for S&P 500 constituents via yfinance.
- Chunked, concurrent fetching via PriceLoader for massive ticker universes.
- Manage point-in-time universe snapshots to avoid look-ahead bias.
- Cache raw and processed data to disk (Parquet / pickle fallback) to limit API calls.

ABF extensions (Phase A2+):
- Expose shock-date series consumed by harbor.abf.q1.
- Provide rolling constituent windows for crowding analysis (ABF Q2).

Quick start (module-level helpers)
-----------------------------------
    from harbor.data import load_sp500_tickers, load_sp500_prices

    tickers = load_sp500_tickers(as_of="2020-01-01")
    prices  = load_sp500_prices(tickers, start="2018-01-01", end="2020-01-01")

Bulk / repeated use (PriceLoader)
----------------------------------
    from harbor.data import configure, PriceLoader

    # Option A — configure the module-level default once at startup
    configure(chunk_size=50, max_workers=16)

    # Option B — instantiate your own loader with custom settings
    loader = PriceLoader(chunk_size=50, max_workers=16, max_retries=3)
    prices = loader.fetch(tickers, start="2018-01-01", end="2023-01-01")
"""

from harbor.data.load_prices import (
    PriceLoader,
    configure,
    get_default_loader,
    load_risk_free_rate,
    load_sp500_prices,
    load_sp500_tickers,
)

__all__ = [
    # Class
    "PriceLoader",
    # Configuration
    "configure",
    "get_default_loader",
    # Data loaders
    "load_sp500_tickers",
    "load_sp500_prices",
    "load_risk_free_rate",
]
