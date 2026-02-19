"""
harbor.portfolio.construction — Portfolio construction and allocation logic.

ABF mapping: regime-aware position sizing (ABF Q1/Q2 integration).

Planned functionality (Phase H1):
- Mean-variance, risk-parity, and HRP allocation interfaces.
- Constraint handling (long-only, leverage, turnover limits).

ABF extensions (Phase H4):
- Shock-aware position sizing: reduce exposure when vol-control pressure proxy
  is elevated (informed by ABF Q1 results).
- Crowding detection module: flag high signal-similarity regimes and adjust
  diversification assumptions (informed by ABF Q2 results).
- Baseline vs regime-aware mode toggling for backtest comparisons.
"""


def hrp_weights(cov_matrix):
    """Compute Hierarchical Risk Parity weights.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Asset covariance matrix.

    Returns
    -------
    pd.Series
        HRP portfolio weights.
    """
    raise NotImplementedError("Phase H1 — implement HRP in harbor.risk then wire here")


def regime_aware_position_size(base_weights, shock_proxy, crowding_proxy=None,
                                shock_scale=0.5):
    """Scale position sizes down when shock or crowding proxies are elevated.

    Parameters
    ----------
    base_weights : pd.Series
        Baseline portfolio weights.
    shock_proxy : float
        Current vol-control pressure proxy level (0–1 normalized).
    crowding_proxy : float, optional
        Current signal-similarity/crowding proxy level (0–1 normalized).
    shock_scale : float
        Minimum weight multiplier when proxy is maximal (default: 0.5).

    Returns
    -------
    pd.Series
        Adjusted portfolio weights.
    """
    raise NotImplementedError("Phase H4 — implement after ABF Q1/Q2 results")
