"""Monte Carlo return simulation and portfolio VaR/CVaR utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

SimulationMethod = Literal["normal", "student_t", "factor"]

import numpy as np
import pandas as pd

from hangar.risk.covariance import estimate_covariance


@dataclass(frozen=True)
class VarCvarResult:
    """Container for Monte Carlo risk estimates."""

    alpha: float
    var: float
    cvar: float
    expected_return: float


def simulate_multivariate_returns(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    *,
    n_sims: int = 10_000,
    horizon: int = 21,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate Monte Carlo return paths from a multivariate normal model.

    Returns an array with shape ``(n_sims, horizon, n_assets)``.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    mu = mean_returns.to_numpy(dtype=float)
    cov = cov_matrix.loc[mean_returns.index, mean_returns.index].to_numpy(dtype=float)

    rng = np.random.default_rng(seed=random_state)
    draws = rng.multivariate_normal(mu, cov, size=(n_sims, horizon), check_valid="ignore")
    return draws


def simulate_student_t_returns(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    *,
    df: int = 5,
    n_sims: int = 10_000,
    horizon: int = 21,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate Monte Carlo return paths from a multivariate Student-t model.

    Uses the standard construction: generate multivariate normal samples and
    scale by an independent chi-squared draw to obtain Student-t marginals with
    ``df`` degrees of freedom.

    Parameters
    ----------
    mean_returns : pd.Series
        Expected daily returns per asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    df : int, default 5
        Degrees of freedom for the Student-t distribution.  Must be > 2.
    n_sims : int, default 10_000
        Number of simulation paths.
    horizon : int, default 21
        Number of daily steps per path.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_sims, horizon, n_assets)``.
    """
    if df <= 2:
        raise ValueError("df must be > 2")
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    mu = mean_returns.to_numpy(dtype=float)
    cov = cov_matrix.loc[mean_returns.index, mean_returns.index].to_numpy(dtype=float)

    rng = np.random.default_rng(seed=random_state)

    # Multivariate normal draws (zero-mean, unit scale)
    normal_draws = rng.multivariate_normal(
        np.zeros(len(mu)), cov, size=(n_sims, horizon), check_valid="ignore"
    )

    # Independent chi-squared scaling factor
    chi2 = rng.chisquare(df, size=(n_sims, horizon))
    scaling = np.sqrt(df / chi2)[..., np.newaxis]  # (n_sims, horizon, 1)

    draws = mu + normal_draws / scaling
    return draws


def simulate_factor_returns(
    factor_loadings: pd.DataFrame,
    factor_cov: pd.DataFrame,
    idio_var: pd.Series,
    *,
    n_sims: int = 10_000,
    horizon: int = 21,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate Monte Carlo return paths from a factor model.

    Systematic returns are computed as ``loadings @ factor_draws`` and
    independent idiosyncratic noise is added per asset.

    Parameters
    ----------
    factor_loadings : pd.DataFrame
        Factor loading matrix with shape ``(n_assets, n_factors)``.
    factor_cov : pd.DataFrame
        Factor covariance matrix with shape ``(n_factors, n_factors)``.
    idio_var : pd.Series
        Per-asset idiosyncratic variance.
    n_sims : int, default 10_000
        Number of simulation paths.
    horizon : int, default 21
        Number of daily steps per path.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_sims, horizon, n_assets)``.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    loadings = factor_loadings.to_numpy(dtype=float)  # (n_assets, n_factors)
    f_cov = factor_cov.to_numpy(dtype=float)           # (n_factors, n_factors)
    idio = idio_var.to_numpy(dtype=float)               # (n_assets,)

    n_assets = loadings.shape[0]
    n_factors = loadings.shape[1]

    rng = np.random.default_rng(seed=random_state)

    # Factor draws: (n_sims, horizon, n_factors)
    factor_draws = rng.multivariate_normal(
        np.zeros(n_factors), f_cov, size=(n_sims, horizon), check_valid="ignore"
    )

    # Systematic component: (n_sims, horizon, n_assets)
    systematic = np.einsum("ijk,lk->ijl", factor_draws, loadings)

    # Idiosyncratic component: independent normal per asset
    idio_std = np.sqrt(idio)
    idio_draws = rng.standard_normal(size=(n_sims, horizon, n_assets)) * idio_std

    return systematic + idio_draws


def simulate_returns(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    *,
    method: SimulationMethod = "normal",
    n_sims: int = 10_000,
    horizon: int = 21,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Unified dispatch for return simulation.

    Parameters
    ----------
    mean_returns : pd.Series
        Expected daily returns per asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    method : {"normal", "student_t", "factor"}
        Simulation method to use.
    n_sims : int, default 10_000
        Number of simulation paths.
    horizon : int, default 21
        Number of daily steps per path.
    random_state : int or None
        Seed for reproducibility.
    **kwargs
        Extra keyword arguments forwarded to the underlying engine (e.g.
        ``df`` for ``"student_t"``).

    Returns
    -------
    np.ndarray
        Array with shape ``(n_sims, horizon, n_assets)``.
    """
    if method == "normal":
        return simulate_multivariate_returns(
            mean_returns, cov_matrix, n_sims=n_sims, horizon=horizon,
            random_state=random_state,
        )
    elif method == "student_t":
        df = kwargs.get("df", 5)
        return simulate_student_t_returns(
            mean_returns, cov_matrix, df=df, n_sims=n_sims, horizon=horizon,
            random_state=random_state,
        )
    elif method == "factor":
        raise ValueError(
            "Use simulate_factor_returns directly (different signature)."
        )
    else:
        raise ValueError(f"Unknown simulation method: {method!r}")


def portfolio_var_cvar(
    weights: pd.Series,
    simulated_returns: np.ndarray,
    *,
    alpha: float = 0.95,
) -> VarCvarResult:
    """Compute VaR/CVaR from simulated return paths for a weighted portfolio."""
    if simulated_returns.ndim != 3:
        raise ValueError("simulated_returns must be a 3D array (n_sims, horizon, n_assets).")

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    w = weights.to_numpy(dtype=float)
    if simulated_returns.shape[2] != w.shape[0]:
        raise ValueError("weights length does not match simulated asset dimension.")

    portfolio_daily = np.tensordot(simulated_returns, w, axes=([2], [0]))
    horizon_returns = np.prod(1.0 + portfolio_daily, axis=1) - 1.0

    losses = -horizon_returns
    var_cutoff = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_cutoff]

    return VarCvarResult(
        alpha=alpha,
        var=var_cutoff,
        cvar=float(np.mean(tail_losses)),
        expected_return=float(np.mean(horizon_returns)),
    )


def monte_carlo_var_cvar_from_history(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    covariance_method: str = "ledoit_wolf",
    n_sims: int = 10_000,
    horizon: int = 21,
    alpha: float = 0.95,
    random_state: Optional[int] = None,
    simulation_method: SimulationMethod = "normal",
    simulation_kwargs: Optional[Dict[str, Any]] = None,
) -> VarCvarResult:
    """Estimate portfolio VaR/CVaR from historical returns via Monte Carlo.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical daily returns.
    weights : pd.Series
        Portfolio weights keyed by asset.
    covariance_method : str, default "ledoit_wolf"
        Covariance estimator forwarded to :func:`estimate_covariance`.
    n_sims : int, default 10_000
        Number of simulation paths.
    horizon : int, default 21
        Number of daily steps per path.
    alpha : float, default 0.95
        Confidence level for VaR/CVaR.
    random_state : int or None
        Seed for reproducibility.
    simulation_method : SimulationMethod, default "normal"
        Simulation engine to use (forwarded to :func:`simulate_returns`).
    simulation_kwargs : dict or None
        Extra keyword arguments forwarded to the simulation engine (e.g.
        ``{"df": 4}`` for Student-t).
    """
    if simulation_kwargs is None:
        simulation_kwargs = {}

    aligned = returns[weights.index].dropna(how="any")
    if aligned.empty:
        raise ValueError("No aligned return history available for the provided weights.")

    mean = aligned.mean()
    cov = estimate_covariance(aligned, method=covariance_method, annualization=1)

    sims = simulate_returns(
        mean,
        cov,
        method=simulation_method,
        n_sims=n_sims,
        horizon=horizon,
        random_state=random_state,
        **simulation_kwargs,
    )

    return portfolio_var_cvar(weights, sims, alpha=alpha)
