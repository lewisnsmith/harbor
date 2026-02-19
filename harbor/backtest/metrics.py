"""
harbor.backtest.metrics — Performance and risk metrics for backtests.

ABF mapping: persistence/reversal metrics (ABF Q1).

Planned functionality (Phase H1):
- Sharpe ratio, Sortino ratio, Calmar ratio.
- Maximum drawdown and drawdown duration.
- Cumulative abnormal return (CAR) for event-study windows.
- Turnover and transaction cost accounting.

ABF extensions (Phase A2+):
- Sign-preserving return day counts (persistence metric).
- Sign-flip probability at 1-month horizon (reversal metric).
- Regime-conditional metric comparison (shock vs non-shock).
"""


def sharpe_ratio(returns, risk_free=0.0, annualization=252):
    """Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio return series.
    risk_free : float
        Daily risk-free rate (default: 0).
    annualization : int
        Trading days per year (default: 252).

    Returns
    -------
    float
    """
    raise NotImplementedError("Phase H1 — implement core backtest metrics")


def max_drawdown(returns):
    """Compute maximum peak-to-trough drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio return series.

    Returns
    -------
    float
        Maximum drawdown as a negative fraction.
    """
    raise NotImplementedError("Phase H1 — implement core backtest metrics")


def cumulative_abnormal_return(returns, event_dates, horizon=21):
    """Compute cumulative abnormal returns around event dates.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    event_dates : array-like
        Dates of shock/event occurrences.
    horizon : int
        Forward-looking window in trading days (default: 21 ≈ 1 month).

    Returns
    -------
    pd.DataFrame
        CARs indexed by event date.
    """
    raise NotImplementedError("Phase A2 — implement for ABF Q1 event study")
