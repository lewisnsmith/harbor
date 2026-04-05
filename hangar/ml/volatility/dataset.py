"""harbor.ml.volatility.dataset — Dataset construction for volatility forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class VolDatasetConfig:
    """Configuration for volatility dataset construction."""

    seq_len: int = 60
    target_horizon: int = 1
    rv_window: int = 21
    features: Tuple[str, ...] = ("returns", "realized_vol")


class VolatilityDataset(Dataset):
    """PyTorch Dataset for volatility forecasting on a single asset.

    Each sample is ``(X, y)`` where:
        X: tensor of shape ``(seq_len, n_features)``
        y: tensor of shape ``(1,)`` — target realized vol at ``t + target_horizon``

    Parameters
    ----------
    returns
        Daily return series for a single asset. ``pd.Series`` with DatetimeIndex.
    config
        Dataset configuration controlling sequence length, features, etc.
    """

    def __init__(
        self,
        returns: pd.Series,
        config: Optional[VolDatasetConfig] = None,
    ) -> None:
        if config is None:
            config = VolDatasetConfig()
        self._config = config

        features = build_features(returns, rv_window=config.rv_window)
        target = build_target(
            returns, rv_window=config.rv_window, horizon=config.target_horizon
        )

        # Align features and target on common index (drop NaN from both)
        common_idx = features.index.intersection(target.dropna().index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]

        # We need seq_len consecutive samples, so the effective dataset starts
        # at index seq_len - 1.
        self._features = torch.tensor(
            features.values, dtype=torch.float32
        )
        self._targets = torch.tensor(
            target.values, dtype=torch.float32
        )
        self._dates = common_idx
        self._seq_len = config.seq_len

    def __len__(self) -> int:
        n = len(self._features) - self._seq_len
        return max(n, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._features[idx : idx + self._seq_len]  # (seq_len, n_features)
        y = self._targets[idx + self._seq_len - 1].unsqueeze(0)  # (1,)
        return x, y

    @property
    def n_features(self) -> int:
        """Number of input features per timestep."""
        return int(self._features.shape[1])

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Dates corresponding to each sample's target observation."""
        start = self._seq_len - 1
        return self._dates[start : start + len(self)]


def build_features(
    returns: pd.Series,
    *,
    rv_window: int = 21,
) -> pd.DataFrame:
    """Construct feature matrix from a return series.

    Features:
    - ``returns``: raw daily return
    - ``realized_vol``: rolling standard deviation over ``rv_window`` days

    Parameters
    ----------
    returns
        Daily return series (single asset).
    rv_window
        Window for rolling realized volatility.

    Returns
    -------
    pd.DataFrame
        Feature matrix with DatetimeIndex, columns = feature names.
        Rows with NaN (from rolling window warmup) are dropped.
    """
    df = pd.DataFrame(
        {
            "returns": returns,
            "realized_vol": returns.rolling(rv_window).std(),
        },
        index=returns.index,
    )
    return df.dropna()


def build_target(
    returns: pd.Series,
    *,
    rv_window: int = 21,
    horizon: int = 1,
) -> pd.Series:
    """Construct the forecast target: realized vol at t + horizon.

    Parameters
    ----------
    returns
        Daily return series (single asset).
    rv_window
        Window for realized vol computation.
    horizon
        Number of days ahead for the target.

    Returns
    -------
    pd.Series
        Target series. Rows that cannot be computed (tail) are NaN.
    """
    rv = returns.rolling(rv_window).std()
    target = rv.shift(-horizon)
    target.name = "target_rv"
    return target


def create_walk_forward_splits(
    dataset_len: int,
    *,
    min_train_size: int = 504,
    val_size: int = 63,
    step_size: int = 63,
    expanding: bool = True,
) -> List[Tuple[range, range]]:
    """Generate (train_indices, val_indices) splits for walk-forward evaluation.

    Parameters
    ----------
    dataset_len
        Total number of samples in the dataset.
    min_train_size
        Minimum number of training samples in the first fold.
    val_size
        Number of validation samples per fold.
    step_size
        Number of samples to advance the window between folds.
    expanding
        If True, training window grows over time. If False, it rolls
        with fixed size ``min_train_size``.

    Returns
    -------
    List[Tuple[range, range]]
        List of ``(train_range, val_range)`` pairs.
    """
    splits: List[Tuple[range, range]] = []
    fold_start = min_train_size

    while fold_start + val_size <= dataset_len:
        val_range = range(fold_start, fold_start + val_size)
        if expanding:
            train_range = range(0, fold_start)
        else:
            train_start = max(0, fold_start - min_train_size)
            train_range = range(train_start, fold_start)
        splits.append((train_range, val_range))
        fold_start += step_size

    return splits
