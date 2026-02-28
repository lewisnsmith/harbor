"""harbor.abf.q1.analysis — Core local projection analysis for ABF Q1.

Provides forward-return computation, rolling autocorrelation by regime,
control-variable construction, and Newey-West local projections to test
whether vol shocks produce short-horizon persistence followed by
medium-horizon reversal.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_forward_returns(
    returns: pd.Series,
    horizons: List[int] = None,
) -> pd.DataFrame:
    """Compute cumulative forward returns at each horizon.

    Parameters
    ----------
    returns
        Daily return series with DatetimeIndex.
    horizons
        Forward horizons in trading days. Defaults to ``[1, 5, 21]``.

    Returns
    -------
    pd.DataFrame
        Columns ``fwd_1``, ``fwd_5``, ``fwd_21`` (or as specified), indexed
        by the date at which the forward window *starts*.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    result = {}
    for h in horizons:
        # Cumulative return over the next h days: prod(1 + r_{t+1..t+h}) - 1
        fwd = (1 + returns).rolling(window=h).apply(np.prod, raw=True).shift(-h) - 1
        result[f"fwd_{h}"] = fwd

    return pd.DataFrame(result, index=returns.index)


def compute_return_autocorrelation(
    returns: pd.Series,
    shock_mask: pd.Series,
    window: int = 5,
) -> pd.DataFrame:
    """Rolling autocorrelation split by shock / non-shock regime.

    Parameters
    ----------
    returns
        Daily return series.
    shock_mask
        Boolean series (True on shock dates).
    window
        Rolling window for lag-1 autocorrelation.

    Returns
    -------
    pd.DataFrame
        Columns ``autocorr``, ``regime`` (``"shock"`` or ``"normal"``).
    """
    autocorr = returns.rolling(window).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    regime = shock_mask.reindex(returns.index, fill_value=False)
    labels = regime.map({True: "shock", False: "normal"})
    return pd.DataFrame(
        {"autocorr": autocorr, "regime": labels},
        index=returns.index,
    )


def build_control_matrix(
    returns: pd.Series,
    market_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Construct the control-variable matrix for local projections.

    Controls:
    - ``baseline_volatility``: 21-day rolling standard deviation
    - ``liquidity_proxy``: 21-day rolling mean of absolute returns
    - ``market_beta``: 126-day rolling OLS beta vs market (if provided)
    - ``dow_mon`` … ``dow_fri``: day-of-week dummies

    Parameters
    ----------
    returns
        Daily return series.
    market_returns
        Market (benchmark) daily returns. If ``None``, market_beta is omitted.

    Returns
    -------
    pd.DataFrame
        Control variables aligned to ``returns.index``.
    """
    controls: Dict[str, pd.Series] = {}

    controls["baseline_volatility"] = returns.rolling(21).std()
    controls["liquidity_proxy"] = returns.abs().rolling(21).mean()

    if market_returns is not None:
        mkt = market_returns.reindex(returns.index)
        # Rolling 126-day beta
        xy_cov = returns.rolling(126).cov(mkt)
        mkt_var = mkt.rolling(126).var()
        beta = xy_cov / mkt_var.replace(0, np.nan)
        controls["market_beta"] = beta

    # Day-of-week dummies
    dow = pd.get_dummies(returns.index.dayofweek, prefix="dow", dtype=float)
    dow.index = returns.index
    for col in dow.columns:
        controls[col] = dow[col]

    df = pd.DataFrame(controls, index=returns.index)
    return df


def fit_local_projection(
    returns: pd.Series,
    shocks: pd.Series,
    vol_proxy: pd.Series,
    controls: pd.DataFrame,
    horizon: int,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run a single-horizon local projection with Newey-West HAC.

    Model::

        r_{t+h} = a + b_h * Shock_t + c_h * Shock_t * VolProxy_t
                  + Gamma * X_t + e_{t+h}

    Parameters
    ----------
    returns
        Daily return series.
    shocks
        Boolean shock indicator.
    vol_proxy
        Vol-control pressure proxy (continuous).
    controls
        Control variables from ``build_control_matrix``.
    horizon
        Forward horizon in trading days.

    Returns
    -------
    statsmodels RegressionResults
        OLS with Newey-West HAC standard errors (maxlags = max(h, 5)).
    """
    # Dependent variable: cumulative forward return at horizon h
    y = (1 + returns).rolling(window=horizon).apply(np.prod, raw=True).shift(-horizon) - 1

    shock_float = shocks.astype(float).reindex(returns.index, fill_value=0.0)
    proxy = vol_proxy.reindex(returns.index)
    interaction = shock_float * proxy

    X = controls.copy()
    X["shock"] = shock_float
    X["shock_x_vol_proxy"] = interaction
    X = sm.add_constant(X, has_constant="add")

    # Align and drop NaN
    combined = pd.concat([y.rename("y"), X], axis=1).dropna()
    if combined.shape[0] < X.shape[1] + 5:
        raise ValueError(f"Insufficient observations ({combined.shape[0]}) for horizon {horizon}.")

    y_clean = combined["y"]
    X_clean = combined.drop(columns=["y"])

    maxlags = max(horizon, 5)
    model = sm.OLS(y_clean, X_clean)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return result


def fit_local_projections(
    returns: pd.Series,
    shocks: pd.Series,
    vol_proxy: pd.Series,
    controls: pd.DataFrame,
    horizons: List[int] = None,
) -> Dict[int, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run local projections across multiple horizons.

    Parameters
    ----------
    returns, shocks, vol_proxy, controls
        See ``fit_local_projection``.
    horizons
        List of forward horizons. Defaults to ``[1, 5, 21]``.

    Returns
    -------
    dict
        Mapping ``horizon -> RegressionResults``.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    results = {}
    for h in horizons:
        results[h] = fit_local_projection(returns, shocks, vol_proxy, controls, h)
    return results
