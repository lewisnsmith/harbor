"""
harbor.data.load_prices — Fetch and cache adjusted price data from yfinance.

Planned functionality (Phase H1):
- Download adjusted daily close prices for a list of tickers.
- Scrape or read a static mapping of S&P 500 constituents with their
  approximate inclusion dates (Wikipedia / Slickcharts) so callers can
  build point-in-time universes.
- Fetch a risk-free rate proxy (e.g. ^IRX 13-week T-bill) aligned to
  the same date range.
- Persist raw downloads to a local Parquet cache under data/ so
  repeated runs do not hit the network.

Dependencies: yfinance, pandas, pathlib (stdlib)
Consumed by: harbor.risk, harbor.backtest, harbor.abf.q1
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sp500_tickers(
    as_of: Optional[Union[str, datetime.date]] = None,
) -> list[str]:
    """Return S&P 500 constituent tickers valid on a given date.

    Parameters
    ----------
    as_of:
        Date for the point-in-time universe snapshot.  Defaults to today
        when ``None``, which returns the *current* constituent list without
        look-ahead protection.

    Returns
    -------
    list[str]
        Sorted list of uppercase ticker symbols (e.g. ``["AAPL", "MSFT", ...]``).

    Raises
    ------
    ValueError
        If ``as_of`` precedes the earliest available universe date.
    NotImplementedError
        Until Phase H1 implementation is complete.
    """
    raise NotImplementedError("Phase H1 — implement S&P 500 universe lookup")


def load_sp500_prices(
    tickers: Sequence[str],
    start: Union[str, datetime.date],
    end: Union[str, datetime.date],
    *,
    adjusted: bool = True,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download (or load from cache) daily prices for the given tickers.

    Parameters
    ----------
    tickers:
        Ticker symbols to fetch.  Symbols not found on yfinance are dropped
        with a warning rather than raising.
    start:
        Inclusive start date, e.g. ``"2010-01-01"`` or a ``datetime.date``.
    end:
        Inclusive end date.
    adjusted:
        When ``True`` (default), use split- and dividend-adjusted closes.
        When ``False``, return raw (unadjusted) closes.
    cache_dir:
        Directory for Parquet cache files.  Defaults to ``./data/cache/``
        relative to the repo root when ``None``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with a ``pd.DatetimeIndex`` and one column per
        ticker.  Missing values (tickers absent on a given date) are
        represented as ``NaN``.

    Raises
    ------
    ValueError
        If ``start`` >= ``end``.
    NotImplementedError
        Until Phase H1 implementation is complete.
    """
    raise NotImplementedError("Phase H1 — implement price loader with yfinance + Parquet cache")


def load_risk_free_rate(
    start: Union[str, datetime.date],
    end: Union[str, datetime.date],
    *,
    proxy_ticker: str = "^IRX",
    annualized: bool = True,
) -> pd.Series:
    """Fetch a daily risk-free rate proxy aligned to the requested date range.

    Parameters
    ----------
    start:
        Inclusive start date.
    end:
        Inclusive end date.
    proxy_ticker:
        yfinance symbol for the rate proxy.  Defaults to ``"^IRX"``
        (13-week US T-bill annualised yield, expressed as a percentage).
    annualized:
        When ``True`` (default), return the rate as an annualised decimal
        (e.g. 0.05 for 5 %).  When ``False``, convert to a daily rate
        (``annualized / 252``).

    Returns
    -------
    pd.Series
        Daily rate values with a ``pd.DatetimeIndex``, forward-filled over
        market holidays.

    Raises
    ------
    NotImplementedError
        Until Phase H1 implementation is complete.
    """
    raise NotImplementedError("Phase H1 — implement risk-free rate loader")
