"""harbor.ml.volatility.baselines — Classical volatility forecasting baselines.

Implements GARCH(1,1), EWMA, and rolling historical volatility as baselines
for benchmarking the LSTM/GRU forecasters (H3 validation Gate 1).

These use only numpy/pandas (no arch package required) to keep dependencies
minimal. The GARCH implementation uses maximum likelihood via scipy.optimize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class BaselineResult:
    """Container for baseline forecast evaluation."""

    name: str
    forecasts: pd.Series
    actuals: pd.Series
    metrics: Dict[str, float]


def ewma_volatility(
    returns: pd.Series,
    *,
    lam: float = 0.94,
    horizon: int = 1,
    annualize: bool = False,
) -> pd.Series:
    """Exponentially weighted moving average volatility forecast.

    Parameters
    ----------
    returns
        Daily return series.
    lam
        Decay factor (RiskMetrics default = 0.94).
    horizon
        Forecast horizon in days (scales by sqrt(horizon)).
    annualize
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        EWMA volatility forecast (one-step-ahead).
    """
    var = returns.ewm(alpha=1 - lam, adjust=False).var()
    sigma = np.sqrt(var * horizon)
    if annualize:
        sigma = sigma * np.sqrt(252)
    return sigma.rename("ewma_vol")


def rolling_volatility(
    returns: pd.Series,
    *,
    window: int = 21,
    horizon: int = 1,
    annualize: bool = False,
) -> pd.Series:
    """Rolling historical standard deviation as naive vol forecast.

    Parameters
    ----------
    returns
        Daily return series.
    window
        Lookback window in days.
    horizon
        Forecast horizon (scales by sqrt(horizon)).
    annualize
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        Rolling vol estimate.
    """
    sigma = returns.rolling(window).std() * np.sqrt(horizon)
    if annualize:
        sigma = sigma * np.sqrt(252)
    return sigma.rename("rolling_vol")


def _garch11_loglik(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1).

    Parameters: [omega, alpha, beta]
    Constraint: alpha + beta < 1, all > 0
    """
    omega, alpha, beta = params
    n = len(returns)

    # Initialize variance at unconditional level
    var_t = omega / max(1 - alpha - beta, 1e-6)
    loglik = 0.0

    for t in range(n):
        if var_t < 1e-12:
            var_t = 1e-12
        loglik += -0.5 * (np.log(2 * np.pi) + np.log(var_t) + returns[t] ** 2 / var_t)
        var_t = omega + alpha * returns[t] ** 2 + beta * var_t

    return -loglik  # minimize negative log-likelihood


def fit_garch11(
    returns: pd.Series,
    *,
    omega0: float = 1e-6,
    alpha0: float = 0.05,
    beta0: float = 0.90,
) -> Dict[str, float]:
    """Fit GARCH(1,1) via maximum likelihood.

    Parameters
    ----------
    returns
        Daily return series (de-meaned recommended).
    omega0, alpha0, beta0
        Initial parameter guesses.

    Returns
    -------
    dict
        Keys: ``omega``, ``alpha``, ``beta``, ``persistence``, ``unconditional_var``.
    """
    r = returns.dropna().values

    x0 = np.array([omega0, alpha0, beta0])
    bounds = [(1e-10, 1.0), (1e-10, 0.9999), (1e-10, 0.9999)]
    constraints = {"type": "ineq", "fun": lambda p: 0.9999 - p[1] - p[2]}

    result = minimize(
        _garch11_loglik,
        x0,
        args=(r,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    omega, alpha, beta = result.x
    persistence = alpha + beta
    uncond_var = omega / max(1 - persistence, 1e-8)

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "unconditional_var": uncond_var,
        "converged": result.success,
    }


def garch11_forecast(
    returns: pd.Series,
    *,
    omega: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    horizon: int = 1,
    annualize: bool = False,
) -> pd.Series:
    """One-step-ahead GARCH(1,1) volatility forecast.

    If parameters are not provided, fits GARCH(1,1) to the full series first.

    Parameters
    ----------
    returns
        Daily return series.
    omega, alpha, beta
        GARCH(1,1) parameters. If None, estimated via MLE.
    horizon
        Forecast horizon (scales by sqrt(horizon)).
    annualize
        If True, multiply by sqrt(252).

    Returns
    -------
    pd.Series
        GARCH(1,1) conditional volatility (one-step-ahead).
    """
    r = returns.dropna()

    if omega is None or alpha is None or beta is None:
        params = fit_garch11(r)
        omega = params["omega"]
        alpha = params["alpha"]
        beta = params["beta"]

    # Generate conditional variance series
    var_series = np.zeros(len(r))
    var_series[0] = omega / max(1 - alpha - beta, 1e-6)

    r_vals = r.values
    for t in range(1, len(r)):
        var_series[t] = omega + alpha * r_vals[t - 1] ** 2 + beta * var_series[t - 1]

    sigma = pd.Series(np.sqrt(var_series * horizon), index=r.index, name="garch_vol")
    if annualize:
        sigma = sigma * np.sqrt(252)
    return sigma


def evaluate_forecast(
    forecasts: pd.Series,
    actuals: pd.Series,
) -> Dict[str, float]:
    """Compute standard volatility forecast evaluation metrics.

    Parameters
    ----------
    forecasts
        Predicted volatility series.
    actuals
        Realized volatility series (e.g., absolute returns or realized vol).

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``qlike``, ``directional_accuracy``.
    """
    aligned = pd.DataFrame({"forecast": forecasts, "actual": actuals}).dropna()
    if aligned.empty:
        return {"rmse": np.nan, "mae": np.nan, "qlike": np.nan, "directional_accuracy": np.nan}

    f = aligned["forecast"].values
    a = aligned["actual"].values

    residuals = f - a
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    # QLIKE: quasi-likelihood loss for variance forecasts
    # QLIKE = mean(actual/forecast + log(forecast) - 1 - log(actual))
    f_safe = np.clip(f, 1e-12, None)
    a_safe = np.clip(a, 1e-12, None)
    qlike = float(np.mean(a_safe / f_safe + np.log(f_safe) - 1 - np.log(a_safe)))

    # Directional accuracy: did forecast direction change match actual?
    if len(f) > 1:
        f_dir = np.diff(f) > 0
        a_dir = np.diff(a) > 0
        directional_accuracy = float(np.mean(f_dir == a_dir))
    else:
        directional_accuracy = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "qlike": qlike,
        "directional_accuracy": directional_accuracy,
    }


def run_baseline_comparison(
    returns: pd.Series,
    *,
    realized_vol: Optional[pd.Series] = None,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run all baselines and return comparison metrics.

    Parameters
    ----------
    returns
        Daily return series.
    realized_vol
        Realized volatility to compare against. If None, uses absolute returns
        as a proxy (standard in vol forecasting literature).
    horizons
        Forecast horizons to evaluate. Defaults to [1].

    Returns
    -------
    pd.DataFrame
        Rows = baselines, columns = metrics.
    """
    if horizons is None:
        horizons = [1]

    if realized_vol is None:
        realized_vol = returns.abs()

    results = []

    for h in horizons:
        # EWMA
        ewma = ewma_volatility(returns, horizon=h)
        ewma_metrics = evaluate_forecast(ewma, realized_vol)
        ewma_metrics["model"] = "EWMA"
        ewma_metrics["horizon"] = h
        results.append(ewma_metrics)

        # Rolling 21-day
        rolling = rolling_volatility(returns, window=21, horizon=h)
        rolling_metrics = evaluate_forecast(rolling, realized_vol)
        rolling_metrics["model"] = "Rolling_21d"
        rolling_metrics["horizon"] = h
        results.append(rolling_metrics)

        # GARCH(1,1)
        garch = garch11_forecast(returns, horizon=h)
        garch_metrics = evaluate_forecast(garch, realized_vol)
        garch_metrics["model"] = "GARCH(1,1)"
        garch_metrics["horizon"] = h
        results.append(garch_metrics)

    df = pd.DataFrame(results)
    return df.set_index(["model", "horizon"]).sort_index()
