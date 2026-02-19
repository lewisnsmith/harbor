"""
harbor.risk.regime_detection — Regime detection and shock identification.

ABF mapping: shock detection + vol-control proxy (ABF Q1 core).

Planned functionality:
- Identify high-volatility shock regimes (top-5% realized vol changes, VIX jumps).
- Vol-control pressure proxy: risk-parity rebalancing signal, ETF flow reversals.
- Regime-aware covariance switching (vol-state shrinkage).

Used by:
- harbor.abf.q1 for event-study shock windows.
- harbor.backtest in regime-aware mode (ABF integration).
"""


def detect_vol_shocks(returns, threshold_pct=0.95):
    """Identify shock dates where realized vol change exceeds ``threshold_pct`` percentile.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Daily return series.
    threshold_pct : float
        Percentile cutoff for shock classification (default: top 5%).

    Returns
    -------
    pd.Series[bool]
        Boolean mask of shock dates.
    """
    raise NotImplementedError("Phase A2 — implement in harbor.abf.q1 pipeline")
