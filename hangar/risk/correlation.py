"""
hangar.risk.correlation — Correlation modeling and spike detection.

ABF mapping: correlation spike detection (Q2 — Regime Manufacturing, Layer 4 roadmap).

Planned functionality:
- Rolling cross-asset correlation estimation.
- Correlation spike detection (>75th percentile of trailing 12-month distribution).
- Correlation regime classification for regime-aware portfolio construction.

Used by:
- ABF Q2 experiments for crowding/correlation spike analysis.
- hangar.portfolio.construction for diversification assumptions.
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
    raise NotImplementedError("Layer 4 roadmap — implement in ABF Q2 experiment pipeline")
