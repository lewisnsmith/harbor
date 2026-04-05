"""Hierarchical Risk Parity (HRP) allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def hrp_allocation(cov_matrix: pd.DataFrame, *, linkage_method: str = "single") -> pd.Series:
    """Compute long-only HRP weights from a covariance matrix."""
    cov = _validate_covariance(cov_matrix)
    corr = cov_to_corr(cov)

    distances = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    condensed = squareform(distances, checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)

    sorted_indices = _quasi_diagonal(linkage_matrix)
    sorted_labels = [cov.index[i] for i in sorted_indices]

    weights = pd.Series(1.0, index=sorted_labels)
    clusters = [sorted_labels]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        left_cluster = cluster[:split]
        right_cluster = cluster[split:]

        left_var = _cluster_variance(cov, left_cluster)
        right_var = _cluster_variance(cov, right_cluster)

        allocation = 1.0 - (left_var / (left_var + right_var))
        weights[left_cluster] *= allocation
        weights[right_cluster] *= 1.0 - allocation

        clusters.append(left_cluster)
        clusters.append(right_cluster)

    weights = weights.reindex(cov.index)
    weights = weights / weights.sum()
    return weights


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    std = np.clip(std, 1e-12, None)
    denom = np.outer(std, std)
    corr = cov.to_numpy(dtype=float) / denom
    corr = np.clip(corr, -1.0, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def _cluster_variance(cov: pd.DataFrame, cluster_items: list[str]) -> float:
    sub_cov = cov.loc[cluster_items, cluster_items]
    variances = np.clip(np.diag(sub_cov.to_numpy(dtype=float)), 1e-12, None)
    inv_diag = 1.0 / variances
    ivp = inv_diag / inv_diag.sum()
    return float(np.dot(ivp, np.dot(sub_cov.to_numpy(dtype=float), ivp)))


def _quasi_diagonal(linkage_matrix: np.ndarray) -> list[int]:
    linkage_matrix = linkage_matrix.astype(int)
    sort_order = [int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])]
    n_items = linkage_matrix[-1, 3]

    while max(sort_order) >= n_items:
        new_order = []
        for index in sort_order:
            if index < n_items:
                new_order.append(index)
            else:
                left = linkage_matrix[index - int(n_items), 0]
                right = linkage_matrix[index - int(n_items), 1]
                new_order.extend([int(left), int(right)])
        sort_order = new_order

    return sort_order


def _validate_covariance(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a pandas DataFrame.")

    if cov_matrix.empty:
        raise ValueError("cov_matrix is empty.")

    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be square.")

    if not cov_matrix.index.equals(cov_matrix.columns):
        raise ValueError("cov_matrix index and columns must match asset labels.")

    cov = cov_matrix.astype(float)
    if np.isnan(cov.to_numpy()).any():
        raise ValueError("cov_matrix contains NaN values.")

    # Ensure numerical symmetry before downstream linear algebra.
    sym = (cov + cov.T) / 2.0
    return sym
