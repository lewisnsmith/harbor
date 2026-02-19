"""
harbor.risk.correlation — Correlation modeling and spike detection.

ABF mapping: correlation spike detection (ABF Q2 core).

Planned functionality:
- Rolling cross-asset correlation estimation.
- Correlation spike detection (>75th percentile of trailing 12-month distribution).
- Correlation regime classification for regime-aware portfolio construction.

Used by:
- harbor.abf.q2 for crowding/correlation spike analysis.
- harbor.portfolio.construction for diversification assumptions.
"""


def detect_correlation_spikes(returns, window=252, spike_pct=0.75):
    """Flag periods where cross-asset correlation exceeds the trailing spike threshold.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return panel (assets as columns).
    window : int
        Rolling window in trading days for baseline distribution (default: 252).
    spike_pct : float
        Percentile above which correlation is classified as a spike (default: 75th).

    Returns
    -------
    pd.Series[bool]
        Boolean mask of correlation spike periods.
    """
    raise NotImplementedError("Phase A3 — implement in harbor.abf.q2 pipeline")
