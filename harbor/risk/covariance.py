"""Covariance estimators used by portfolio construction and simulation."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

CovarianceMethod = Literal["sample", "ledoit_wolf", "oas"]


def sample_covariance(returns: pd.DataFrame, *, annualization: int = 252) -> pd.DataFrame:
    """Estimate a sample covariance matrix from asset returns."""
    frame = _validate_returns(returns)
    cov = frame.cov()
    return cov * float(annualization)


def shrinkage_covariance(
    returns: pd.DataFrame,
    *,
    method: Literal["ledoit_wolf", "oas"] = "ledoit_wolf",
    annualization: int = 252,
) -> pd.DataFrame:
    """Estimate a shrinkage covariance matrix (Ledoit-Wolf or OAS)."""
    frame = _validate_returns(returns)
    clean = _impute_missing(frame)

    estimator = LedoitWolf() if method == "ledoit_wolf" else OAS()
    estimator.fit(clean.to_numpy(dtype=float))

    cov = pd.DataFrame(estimator.covariance_, index=clean.columns, columns=clean.columns)
    return cov * float(annualization)


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: CovarianceMethod = "sample",
    annualization: int = 252,
) -> pd.DataFrame:
    """Convenience wrapper for covariance estimation methods."""
    if method == "sample":
        return sample_covariance(returns, annualization=annualization)

    if method in {"ledoit_wolf", "oas"}:
        return shrinkage_covariance(returns, method=method, annualization=annualization)

    raise ValueError(
        f"Unknown covariance method {method!r}. Use one of: sample, ledoit_wolf, oas."
    )


def _validate_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame with assets in columns.")

    if returns.empty:
        raise ValueError("returns DataFrame is empty.")

    frame = returns.astype(float).dropna(how="all")
    if frame.shape[0] < 2:
        raise ValueError("returns must contain at least two rows.")

    usable = frame.dropna(axis=1, how="all")
    if usable.shape[1] < 2:
        raise ValueError("returns must contain at least two assets with observations.")

    return usable


def _impute_missing(frame: pd.DataFrame) -> pd.DataFrame:
    column_means = frame.mean()
    filled = frame.fillna(column_means)
    # If any column is entirely NaN, fall back to zero to keep matrix shape stable.
    return filled.fillna(0.0)
