"""Covariance estimators used by portfolio construction and simulation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

CovarianceMethod = Literal["sample", "ledoit_wolf", "oas", "regime_aware"]


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


def regime_aware_covariance(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    method: Literal["sample", "ledoit_wolf", "oas"] = "ledoit_wolf",
    annualization: int = 252,
) -> dict[str, pd.DataFrame]:
    """Estimate separate covariance matrices per regime, blended by observation count.

    Parameters
    ----------
    returns:
        DataFrame of asset returns (rows=dates, columns=assets).
    regime_labels:
        Series of regime labels aligned to the *returns* index.
        For example ``pd.Series(["high_vol", "low_vol", ...], index=returns.index)``.
    method:
        Covariance estimator to use within each regime.
    annualization:
        Annualization factor applied to every per-regime matrix.

    Returns
    -------
    dict mapping each unique regime label to its annualized covariance DataFrame.
    """
    frame = _validate_returns(returns)

    # Align regime_labels to validated returns index.
    aligned_labels = regime_labels.reindex(frame.index)
    if aligned_labels.isna().all():
        raise ValueError("regime_labels has no overlap with returns index.")

    # Drop rows where the label is missing.
    mask = aligned_labels.notna()
    frame = frame.loc[mask]
    aligned_labels = aligned_labels.loc[mask]

    unique_regimes = aligned_labels.unique()
    if len(unique_regimes) < 1:
        raise ValueError("regime_labels must contain at least one regime.")

    result: dict[str, pd.DataFrame] = {}
    for regime in sorted(unique_regimes, key=str):
        regime_returns = frame.loc[aligned_labels == regime]
        if regime_returns.shape[0] < 2:
            continue
        if method == "sample":
            cov = sample_covariance(regime_returns, annualization=annualization)
        else:
            cov = shrinkage_covariance(regime_returns, method=method, annualization=annualization)
        result[str(regime)] = cov

    if not result:
        raise ValueError("No regime had enough observations (>=2) to estimate a covariance.")

    return result


def expanding_regime_covariance(
    returns: pd.DataFrame,
    vol_threshold_pct: float = 0.8,
    *,
    vol_window: int = 63,
    method: Literal["sample", "ledoit_wolf", "oas"] = "ledoit_wolf",
    annualization: int = 252,
) -> dict[str, pd.DataFrame]:
    """Classify regimes by rolling realized-volatility percentile, then estimate covariances.

    The rolling realized volatility is computed as the cross-sectional mean of
    per-asset rolling standard deviations (window = *vol_window*).  Dates whose
    rolling vol exceeds the *vol_threshold_pct* percentile of the expanding
    distribution are labelled ``"high_vol"``; the rest ``"low_vol"``.

    Parameters
    ----------
    returns:
        DataFrame of asset returns.
    vol_threshold_pct:
        Percentile (0-1) of rolling vol above which a date is ``"high_vol"``.
    vol_window:
        Look-back window for realized vol (business days).
    method:
        Covariance estimator forwarded to :func:`regime_aware_covariance`.
    annualization:
        Annualization factor forwarded to :func:`regime_aware_covariance`.

    Returns
    -------
    dict with keys ``"high_vol"`` and ``"low_vol"`` mapping to covariance DataFrames.
    If one regime has too few observations, only the other key will be present.
    """
    frame = _validate_returns(returns)

    # Per-asset rolling std, then cross-sectional mean -> single vol series.
    rolling_vol = frame.rolling(window=vol_window, min_periods=max(vol_window // 2, 2)).std()
    mean_vol = rolling_vol.mean(axis=1).dropna()

    if mean_vol.empty:
        raise ValueError("Not enough observations to compute rolling volatility.")

    threshold = float(np.percentile(mean_vol.to_numpy(), vol_threshold_pct * 100))

    labels = pd.Series(
        np.where(mean_vol > threshold, "high_vol", "low_vol"),
        index=mean_vol.index,
    )

    return regime_aware_covariance(
        returns,
        labels,
        method=method,
        annualization=annualization,
    )


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

    if method == "regime_aware":
        regimes = expanding_regime_covariance(returns, annualization=annualization)
        # Conservative default: return the high-vol regime matrix when available.
        if "high_vol" in regimes:
            return regimes["high_vol"]
        # Fall back to whichever regime is present.
        return next(iter(regimes.values()))

    raise ValueError(
        f"Unknown covariance method {method!r}. "
        "Use one of: sample, ledoit_wolf, oas, regime_aware."
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
