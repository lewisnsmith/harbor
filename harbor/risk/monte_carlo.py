"""Monte Carlo return simulation and portfolio VaR/CVaR utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from harbor.risk.covariance import estimate_covariance


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
) -> VarCvarResult:
    """Estimate portfolio VaR/CVaR from historical returns via Monte Carlo."""
    aligned = returns[weights.index].dropna(how="any")
    if aligned.empty:
        raise ValueError("No aligned return history available for the provided weights.")

    mean = aligned.mean()
    cov = estimate_covariance(aligned, method=covariance_method, annualization=1)

    sims = simulate_multivariate_returns(
        mean,
        cov,
        n_sims=n_sims,
        horizon=horizon,
        random_state=random_state,
    )

    return portfolio_var_cvar(weights, sims, alpha=alpha)
