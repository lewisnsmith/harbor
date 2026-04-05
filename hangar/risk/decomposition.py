"""Risk decomposition and attribution for HANGAR H2.

Provides marginal/component risk decomposition by asset, factor, and cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_weights_cov(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Ensure *weights* and *cov_matrix* are aligned and conformable."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pd.Series")
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pd.DataFrame")
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square")
    if not cov_matrix.index.equals(cov_matrix.columns):
        raise ValueError("cov_matrix index and columns must match")

    # Align weights to covariance ordering
    missing = set(weights.index) - set(cov_matrix.index)
    if missing:
        raise ValueError(f"Assets in weights not found in cov_matrix: {missing}")
    weights = weights.reindex(cov_matrix.index)
    if weights.isna().any():
        raise ValueError("weights contain NaN after alignment with cov_matrix")
    return weights, cov_matrix


def _portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """Scalar portfolio variance w'Σw."""
    return float(w @ cov @ w)


# ---------------------------------------------------------------------------
# Marginal / component risk
# ---------------------------------------------------------------------------


def marginal_contribution_to_risk(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Compute marginal contribution to risk (MCR) for each asset.

    MCR_i = (Σw)_i / sqrt(w'Σw)

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    cov_matrix : pd.DataFrame
        Covariance matrix with matching index/columns.

    Returns
    -------
    pd.Series
        Marginal contribution to risk for each asset.
    """
    weights, cov_matrix = _validate_weights_cov(weights, cov_matrix)
    w = weights.to_numpy(dtype=float)
    cov = cov_matrix.to_numpy(dtype=float)

    sigma_w = cov @ w
    port_vol = np.sqrt(_portfolio_variance(w, cov))
    if port_vol == 0.0:
        return pd.Series(0.0, index=weights.index, name="mcr")

    mcr = sigma_w / port_vol
    return pd.Series(mcr, index=weights.index, name="mcr")


def component_risk(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Component risk: w_i * MCR_i.

    Component risks sum to total portfolio volatility.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    cov_matrix : pd.DataFrame
        Covariance matrix with matching index/columns.

    Returns
    -------
    pd.Series
        Component risk for each asset.
    """
    weights, cov_matrix = _validate_weights_cov(weights, cov_matrix)
    mcr = marginal_contribution_to_risk(weights, cov_matrix)
    cr = weights * mcr
    cr.name = "component_risk"
    return cr


def percent_risk_contribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.Series:
    """Component risk as a fraction of total portfolio volatility.

    Returns a pd.Series that sums to 1.0.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    cov_matrix : pd.DataFrame
        Covariance matrix with matching index/columns.

    Returns
    -------
    pd.Series
        Percentage risk contribution for each asset (sums to 1.0).
    """
    cr = component_risk(weights, cov_matrix)
    total = cr.sum()
    if total == 0.0:
        prc = pd.Series(0.0, index=cr.index, name="risk_pct")
    else:
        prc = cr / total
        prc.name = "risk_pct"
    return prc


# ---------------------------------------------------------------------------
# Factor risk decomposition
# ---------------------------------------------------------------------------


def factor_risk_decomposition(
    weights: pd.Series,
    factor_loadings: pd.DataFrame,
    factor_cov: pd.DataFrame,
    idio_var: pd.Series,
) -> dict[str, Any]:
    """Decompose portfolio variance into systematic and idiosyncratic parts.

    Uses the model: Σ = B F B' + D  where B = factor_loadings, F = factor_cov,
    D = diag(idio_var).

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    factor_loadings : pd.DataFrame
        Factor loading matrix (n_assets x n_factors).
    factor_cov : pd.DataFrame
        Factor covariance matrix (n_factors x n_factors).
    idio_var : pd.Series
        Per-asset idiosyncratic variance.

    Returns
    -------
    dict
        Keys: ``total_variance``, ``systematic_variance``,
        ``idiosyncratic_variance``, ``systematic_pct``,
        ``factor_contributions`` (pd.Series of variance attributed to each factor).
    """
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pd.Series")
    if not isinstance(factor_loadings, pd.DataFrame):
        raise TypeError("factor_loadings must be a pd.DataFrame")
    if not isinstance(factor_cov, pd.DataFrame):
        raise TypeError("factor_cov must be a pd.DataFrame")
    if not isinstance(idio_var, pd.Series):
        raise TypeError("idio_var must be a pd.Series")

    # Align assets
    assets = weights.index
    missing_in_loadings = set(assets) - set(factor_loadings.index)
    if missing_in_loadings:
        raise ValueError(
            f"Assets in weights not found in factor_loadings: {missing_in_loadings}"
        )
    missing_in_idio = set(assets) - set(idio_var.index)
    if missing_in_idio:
        raise ValueError(
            f"Assets in weights not found in idio_var: {missing_in_idio}"
        )

    B = factor_loadings.loc[assets].to_numpy(dtype=float)  # (n, k)
    factors = factor_loadings.columns
    F = factor_cov.loc[factors, factors].to_numpy(dtype=float)  # (k, k)
    d = idio_var.reindex(assets).to_numpy(dtype=float)  # (n,)
    w = weights.to_numpy(dtype=float)  # (n,)

    # Portfolio factor exposure: θ = B'w  (k,)
    theta = B.T @ w

    # Systematic variance: θ' F θ
    systematic_var = float(theta @ F @ theta)

    # Idiosyncratic variance: w' D w
    idio_total = float(w ** 2 @ d)

    total_var = systematic_var + idio_total

    # Per-factor contribution: decompose θ' F θ into factor-level pieces.
    # Contribution of factor j = θ_j * (F θ)_j
    f_theta = F @ theta  # (k,)
    factor_contribs = theta * f_theta
    factor_contributions = pd.Series(
        factor_contribs, index=factors, name="factor_variance_contribution"
    )

    systematic_pct = systematic_var / total_var if total_var > 0.0 else 0.0

    return {
        "total_variance": total_var,
        "systematic_variance": systematic_var,
        "idiosyncratic_variance": idio_total,
        "systematic_pct": systematic_pct,
        "factor_contributions": factor_contributions,
    }


# ---------------------------------------------------------------------------
# Cluster / sector risk attribution
# ---------------------------------------------------------------------------


def cluster_risk_attribution(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
    cluster_map: dict[str, str],
) -> pd.DataFrame:
    """Attribute portfolio risk to clusters or sectors.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    cov_matrix : pd.DataFrame
        Covariance matrix with matching index/columns.
    cluster_map : dict
        Mapping of asset name -> cluster label.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns: ``cluster``, ``weight``,
        ``risk_contribution``, ``risk_pct``.
    """
    weights, cov_matrix = _validate_weights_cov(weights, cov_matrix)

    missing = set(weights.index) - set(cluster_map.keys())
    if missing:
        raise ValueError(f"Assets missing from cluster_map: {missing}")

    prc = percent_risk_contribution(weights, cov_matrix)
    cr = component_risk(weights, cov_matrix)

    clusters = pd.Series(cluster_map, name="cluster").reindex(weights.index)

    records: list[dict[str, Any]] = []
    for label in sorted(clusters.unique()):
        mask = clusters == label
        records.append(
            {
                "cluster": label,
                "weight": float(weights[mask].sum()),
                "risk_contribution": float(cr[mask].sum()),
                "risk_pct": float(prc[mask].sum()),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


def concentration_metrics(
    weights: pd.Series,
    cov_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """Compute portfolio concentration measures.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    cov_matrix : pd.DataFrame
        Covariance matrix with matching index/columns.

    Returns
    -------
    dict
        Keys: ``herfindahl_weight``, ``herfindahl_risk``,
        ``effective_n_weight``, ``effective_n_risk``,
        ``max_risk_contributor``, ``max_risk_pct``.
    """
    weights, cov_matrix = _validate_weights_cov(weights, cov_matrix)
    prc = percent_risk_contribution(weights, cov_matrix)

    hhi_w = float((weights ** 2).sum())
    hhi_r = float((prc ** 2).sum())

    eff_n_w = 1.0 / hhi_w if hhi_w > 0.0 else float("inf")
    eff_n_r = 1.0 / hhi_r if hhi_r > 0.0 else float("inf")

    max_idx = prc.idxmax()

    return {
        "herfindahl_weight": hhi_w,
        "herfindahl_risk": hhi_r,
        "effective_n_weight": eff_n_w,
        "effective_n_risk": eff_n_r,
        "max_risk_contributor": max_idx,
        "max_risk_pct": float(prc[max_idx]),
    }
