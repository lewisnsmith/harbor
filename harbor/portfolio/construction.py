"""Portfolio construction and allocation logic."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from harbor.risk.hrp import hrp_allocation


def mean_variance_weights(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    *,
    risk_aversion: float = 3.0,
    long_only: bool = True,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    target_leverage: float = 1.0,
) -> pd.Series:
    """Compute mean-variance weights under a budget constraint."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be > 0.")

    labels, mu, cov = _align_inputs(expected_returns, cov_matrix)
    n_assets = len(labels)

    x0 = np.full(n_assets, target_leverage / n_assets)

    if long_only:
        lower, upper = weight_bounds
        bounds = [(lower, upper) for _ in range(n_assets)]
    else:
        bounds = [(-2.0, 2.0) for _ in range(n_assets)]

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - target_leverage}]

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ cov @ weights)
        expected = float(mu @ weights)
        return 0.5 * risk_aversion * variance - expected

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Mean-variance optimization failed: {result.message}")

    return pd.Series(result.x, index=labels, name="mean_variance_weight")


def risk_parity_weights(
    cov_matrix: pd.DataFrame,
    *,
    max_iter: int = 1_000,
    tol: float = 1e-8,
) -> pd.Series:
    """Compute long-only risk parity weights via multiplicative updates."""
    cov = _validate_covariance(cov_matrix).to_numpy(dtype=float)
    labels = cov_matrix.index
    n_assets = len(labels)

    weights = np.full(n_assets, 1.0 / n_assets)

    for _ in range(max_iter):
        portfolio_var = float(weights @ cov @ weights)
        marginal_risk = cov @ weights
        risk_contrib = weights * marginal_risk

        target_contrib = portfolio_var / n_assets
        error = np.max(np.abs(risk_contrib - target_contrib))
        if error < tol:
            break

        safe_contrib = np.where(risk_contrib <= 0.0, 1e-12, risk_contrib)
        weights *= target_contrib / safe_contrib
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()

    weights /= weights.sum()
    return pd.Series(weights, index=labels, name="risk_parity_weight")


def hrp_weights(cov_matrix: pd.DataFrame) -> pd.Series:
    """Compute Hierarchical Risk Parity weights."""
    weights = hrp_allocation(_validate_covariance(cov_matrix))
    weights.name = "hrp_weight"
    return weights


def regime_aware_position_size(
    base_weights: pd.Series,
    shock_proxy: float,
    crowding_proxy: Optional[float] = None,
    shock_scale: float = 0.5,
) -> pd.Series:
    """Scale position sizes down when shock or crowding proxies are elevated.

    The function preserves the relative composition of ``base_weights`` while
    reducing gross exposure. Unallocated weight is interpreted as cash.
    """
    if not (0.0 <= shock_proxy <= 1.0):
        raise ValueError("shock_proxy must be in [0, 1].")
    if crowding_proxy is not None and not (0.0 <= crowding_proxy <= 1.0):
        raise ValueError("crowding_proxy must be in [0, 1].")
    if not (0.0 < shock_scale <= 1.0):
        raise ValueError("shock_scale must be in (0, 1].")

    max_proxy = max(shock_proxy, crowding_proxy if crowding_proxy is not None else 0.0)
    multiplier = 1.0 - max_proxy * (1.0 - shock_scale)
    adjusted = base_weights.astype(float) * multiplier
    adjusted.name = "regime_aware_weight"
    return adjusted


def _align_inputs(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series.")

    cov = _validate_covariance(cov_matrix)
    common = expected_returns.index.intersection(cov.index)
    if len(common) < 2:
        raise ValueError(
            "Need at least two overlapping assets between expected_returns and cov_matrix."
        )

    aligned_returns = expected_returns.loc[common].astype(float)
    aligned_cov = cov.loc[common, common].to_numpy(dtype=float)
    return common, aligned_returns.to_numpy(dtype=float), aligned_cov


def _validate_covariance(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame.")

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square.")

    if not cov_matrix.index.equals(cov_matrix.columns):
        raise ValueError("cov_matrix index and columns must match.")

    cov = cov_matrix.astype(float)
    if np.isnan(cov.to_numpy()).any():
        raise ValueError("cov_matrix contains NaNs.")

    return (cov + cov.T) / 2.0
