"""Regime detection and shock identification for ABF Question 1."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd


def detect_vol_shocks(
    returns: Union[pd.Series, pd.DataFrame],
    threshold_pct: float = 0.95,
    *,
    vol_window: int = 21,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Flag shock dates where realized-vol changes exceed a percentile threshold.

    Parameters
    ----------
    returns
        Daily returns. If a DataFrame is provided, the equal-weighted market
        return is used.
    threshold_pct
        Percentile cutoff for shock classification (0.95 = top 5%).
    vol_window
        Window used for rolling realized volatility.
    min_periods
        Minimum observations for rolling volatility; defaults to ``vol_window``.
    """
    if not (0.0 < threshold_pct < 1.0):
        raise ValueError("threshold_pct must be in (0, 1).")

    if vol_window < 2:
        raise ValueError("vol_window must be >= 2.")

    series = _coerce_market_returns(returns)
    min_obs = min_periods if min_periods is not None else vol_window

    realized_vol = series.rolling(vol_window, min_periods=min_obs).std()
    vol_change = realized_vol.diff().abs()

    cutoff = vol_change.quantile(threshold_pct)
    shocks = (vol_change >= cutoff) & vol_change.notna()
    shocks.name = "vol_shock"
    return shocks.astype(bool)


def vol_control_pressure_proxy(
    returns: Union[pd.Series, pd.DataFrame],
    *,
    short_window: int = 21,
    long_window: int = 126,
) -> pd.Series:
    """Construct a simple vol-control pressure proxy for ABF Q1.

    Proxy definition: ratio of short-horizon to long-horizon realized volatility,
    clipped to [0, 2]. Values above 1 indicate elevated de-risking pressure.
    """
    if short_window <= 1 or long_window <= short_window:
        raise ValueError("Require long_window > short_window > 1.")

    series = _coerce_market_returns(returns)
    short_vol = series.rolling(short_window, min_periods=short_window).std()
    long_vol = series.rolling(long_window, min_periods=long_window).std()

    proxy = (short_vol / long_vol).replace([float("inf"), -float("inf")], pd.NA)
    proxy = proxy.clip(lower=0.0, upper=2.0)
    proxy.name = "vol_control_pressure"
    return proxy


def _coerce_market_returns(returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = returns.astype(float)
    elif isinstance(returns, pd.DataFrame):
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        series = returns.astype(float).mean(axis=1)
    else:
        raise TypeError("returns must be a pandas Series or DataFrame.")

    series = series.dropna().sort_index()
    if series.empty:
        raise ValueError("returns contains no valid observations.")

    return series
