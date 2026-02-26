"""
harbor.backtest.metrics — Performance and risk metrics for backtests.

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
