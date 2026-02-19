"""
harbor.features.crowding — Signal similarity and crowding proxies.

ABF mapping: signal similarity proxies (ABF Q2 core).

Planned functionality (Phase A3):
- Signal correlation across assets (momentum, liquidity, sentiment).
- Momentum crowding proxy: cross-sectional dispersion of returns.
- ETF flow proxies: aggregate inflow/outflow pressure.
- Vol-control exposure proxies: risk-parity deleveraging indicators.

Success criteria (from ABF PRD):
- Proxies predict correlation expansion out-of-sample (AUC > 0.65).
- At least one proxy Granger-causes correlation spikes with 1-5 day lead.
"""


def momentum_crowding_proxy(returns, lookback=252):
    """Compute cross-sectional return dispersion as a momentum crowding proxy.

    Low dispersion = many assets moving together = potential crowding.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily return panel (assets as columns).
    lookback : int
        Rolling window in trading days (default: 252).

    Returns
    -------
    pd.Series
        Time series of cross-sectional return dispersion (std).
    """
    raise NotImplementedError("Phase A3 — implement crowding proxy pipeline")


def signal_similarity(signals, method="correlation"):
    """Measure cross-sectional similarity of trading signals.

    Parameters
    ----------
    signals : pd.DataFrame
        Signal values panel (assets as columns, dates as index).
    method : str
        Similarity method: "correlation" or "cosine" (default: "correlation").

    Returns
    -------
    pd.Series
        Time series of average pairwise signal similarity.
    """
    raise NotImplementedError("Phase A3 — implement crowding proxy pipeline")
