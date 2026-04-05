"""
hangar.backtest.metrics — Performance and risk metrics for backtests.

Provides guarded implementations of standard risk-adjusted return metrics.
All functions require a minimum number of observations and return NaN
(rather than inf) when inputs are degenerate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_returns(returns: pd.Series, *, min_obs: int = 2) -> np.ndarray:
    """Convert *returns* to a clean 1-D float array, raising on bad input."""
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError("returns must be a pandas Series or numpy array")
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_obs:
        raise ValueError(
            f"Need at least {min_obs} finite return observations, got {len(arr)}"
        )
    return arr


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    annualization: int = 252,
    min_obs: int = 2,
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Period (e.g. daily) portfolio return series.
    risk_free : float
        Period risk-free rate (default 0).
    annualization : int
        Periods per year (default 252).
    min_obs : int
        Minimum observations required (default 2).

    Returns
    -------
    float
        Annualized Sharpe ratio, or ``nan`` if volatility is zero.
    """
    arr = _validate_returns(returns, min_obs=min_obs)
    excess = arr - risk_free
    vol = excess.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(annualization))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    annualization: int = 252,
    min_obs: int = 2,
) -> float:
    """Annualized Sortino ratio (downside-deviation denominator).

    Returns ``nan`` when there are no negative excess returns.
    """
    arr = _validate_returns(returns, min_obs=min_obs)
    excess = arr - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    down_vol = downside.std(ddof=1)
    if down_vol == 0:
        return float("nan")
    return float(excess.mean() / down_vol * np.sqrt(annualization))


def max_drawdown(returns: pd.Series, min_obs: int = 2) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction.

    Parameters
    ----------
    returns : pd.Series
        Period portfolio return series.
    min_obs : int
        Minimum observations required.

    Returns
    -------
    float
        Largest drawdown (negative value), e.g. -0.15 means -15%.
    """
    arr = _validate_returns(returns, min_obs=min_obs)
    cumulative = np.cumprod(1 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def calmar_ratio(
    returns: pd.Series,
    annualization: int = 252,
    min_obs: int = 2,
) -> float:
    """Calmar ratio: annualized return / abs(max drawdown).

    Returns ``nan`` when max drawdown is zero.
    """
    arr = _validate_returns(returns, min_obs=min_obs)
    total = np.prod(1 + arr)
    years = len(arr) / annualization
    if years == 0:
        return float("nan")
    cagr = total ** (1 / years) - 1
    mdd = max_drawdown(returns, min_obs=min_obs)
    if mdd == 0:
        return float("nan")
    return float(cagr / abs(mdd))


def win_rate(returns: pd.Series, min_obs: int = 2) -> float:
    """Fraction of periods with positive returns."""
    arr = _validate_returns(returns, min_obs=min_obs)
    return float((arr > 0).mean())


# ---------------------------------------------------------------------------
# Event-study metrics
# ---------------------------------------------------------------------------

def cumulative_abnormal_return(
    returns: pd.Series,
    event_dates: pd.DatetimeIndex,
    horizon: int = 21,
) -> pd.DataFrame:
    """Compute cumulative abnormal returns around event dates.

    For each event date, accumulates forward returns over ``horizon``
    trading days and subtracts the full-sample mean daily return
    (market adjustment) accumulated over the same window.

    Parameters
    ----------
    returns : pd.Series
        Daily return series with DatetimeIndex.
    event_dates : pd.DatetimeIndex
        Dates on which events (e.g. vol shocks) occurred.
    horizon : int
        Number of trading days forward to accumulate returns.

    Returns
    -------
    pd.DataFrame
        One row per usable event date with columns:
        - ``event_date``: the event date
        - ``car``: cumulative abnormal return over the horizon
        - ``cumulative_return``: raw cumulative return
        - ``expected_return``: cumulative expected (mean) return
    """
    if len(returns) == 0 or len(event_dates) == 0:
        return pd.DataFrame(
            columns=["event_date", "car", "cumulative_return", "expected_return"]
        )

    returns = returns.sort_index()
    daily_mean = returns.mean()

    records = []
    dates_array = returns.index

    for event_date in event_dates:
        # Find position of event date (or next available)
        mask = dates_array >= event_date
        if not mask.any():
            continue
        start_loc = int(np.argmax(mask))

        end_loc = start_loc + horizon
        if end_loc > len(returns):
            continue

        window = returns.iloc[start_loc:end_loc]
        cum_ret = float((1 + window).prod() - 1)
        expected_ret = float((1 + daily_mean) ** horizon - 1)
        car = cum_ret - expected_ret

        records.append({
            "event_date": event_date,
            "car": car,
            "cumulative_return": cum_ret,
            "expected_return": expected_ret,
        })

    return pd.DataFrame(records)
