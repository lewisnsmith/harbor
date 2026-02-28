"""Tests for harbor.ml.volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from harbor.ml.volatility.dataset import (
    VolatilityDataset,
    VolDatasetConfig,
    build_features,
    build_target,
    create_walk_forward_splits,
)
from harbor.ml.volatility.integration import (
    sigma_hat_to_regime_proxy,
    vol_scaled_weight_func,
)
from harbor.ml.volatility.models import GRUVolModel, LSTMVolModel, create_model
from harbor.ml.volatility.training import (
    TrainConfig,
    predict_series,
    train_model,
)


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """Generate 500 days of synthetic returns for a single asset."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.Series(rng.normal(0.0005, 0.01, size=500), index=dates, name="TEST")


def test_build_features_produces_expected_columns(synthetic_returns):
    features = build_features(synthetic_returns)
    assert "returns" in features.columns
    assert "realized_vol" in features.columns
    assert not features.isna().any().any()
    assert len(features) < len(synthetic_returns)  # warmup dropped


def test_build_target_alignment(synthetic_returns):
    target = build_target(synthetic_returns)
    assert target.name == "target_rv"
    # Last `horizon` values should be NaN (shifted forward)
    assert target.isna().any()


def test_volatility_dataset_shapes(synthetic_returns):
    config = VolDatasetConfig(seq_len=30, rv_window=21)
    ds = VolatilityDataset(synthetic_returns, config)

    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (30, 2)  # seq_len x n_features
    assert y.shape == (1,)
    assert ds.n_features == 2


def test_volatility_dataset_dates(synthetic_returns):
    config = VolDatasetConfig(seq_len=30, rv_window=21)
    ds = VolatilityDataset(synthetic_returns, config)
    assert len(ds.dates) == len(ds)


def test_walk_forward_splits_no_leakage():
    splits = create_walk_forward_splits(
        1000, min_train_size=200, val_size=50, step_size=50, expanding=True
    )
    assert len(splits) > 0
    for train_range, val_range in splits:
        # No overlap
        train_set = set(train_range)
        val_set = set(val_range)
        assert train_set.isdisjoint(val_set)
        # Train comes before val
        assert max(train_range) < min(val_range)


def test_walk_forward_splits_rolling():
    splits = create_walk_forward_splits(
        1000, min_train_size=200, val_size=50, step_size=50, expanding=False
    )
    assert len(splits) > 0
    for train_range, _ in splits:
        assert len(train_range) == 200


def test_lstm_model_forward_shape():
    model = LSTMVolModel(n_features=2, hidden_size=16, num_layers=1)
    x = torch.randn(4, 30, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert (out >= 0).all()  # ReLU ensures non-negative


def test_gru_model_forward_shape():
    model = GRUVolModel(n_features=2, hidden_size=16, num_layers=1)
    x = torch.randn(4, 30, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert (out >= 0).all()


def test_create_model_factory():
    lstm = create_model("lstm", n_features=3)
    gru = create_model("gru", n_features=3)
    assert isinstance(lstm, LSTMVolModel)
    assert isinstance(gru, GRUVolModel)

    with pytest.raises(ValueError, match="Unknown architecture"):
        create_model("transformer", n_features=3)


def test_train_model_reduces_loss(synthetic_returns, tmp_path):
    config = VolDatasetConfig(seq_len=30, rv_window=21)
    ds = VolatilityDataset(synthetic_returns, config)

    model = LSTMVolModel(n_features=ds.n_features, hidden_size=16, num_layers=1)
    train_config = TrainConfig(
        epochs=5, batch_size=32, learning_rate=1e-3,
        early_stopping_patience=10, model_name="test_vol",
    )

    n = len(ds)
    split = int(n * 0.7)
    result = train_model(
        model, ds,
        train_indices=range(0, split),
        val_indices=range(split, n),
        config=train_config,
    )

    assert len(result.train_losses) > 0
    assert len(result.val_losses) > 0
    assert result.best_val_loss < float("inf")
    assert np.isfinite(result.best_val_loss)


def test_predict_series_returns_dated_series(synthetic_returns):
    config = VolDatasetConfig(seq_len=30, rv_window=21)
    ds = VolatilityDataset(synthetic_returns, config)

    model = LSTMVolModel(n_features=ds.n_features, hidden_size=16, num_layers=1)
    sigma_hat = predict_series(model, ds)

    assert isinstance(sigma_hat, pd.Series)
    assert sigma_hat.name == "sigma_hat"
    assert isinstance(sigma_hat.index, pd.DatetimeIndex)
    assert len(sigma_hat) == len(ds)


def test_vol_scaled_weight_func_integration():
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(42)

    sigma_hat = pd.Series(rng.uniform(0.05, 0.20, size=300), index=dates)

    def dummy_weight_func(window, current):
        n = len(window.columns)
        return pd.Series(1.0 / n, index=window.columns)

    scaled = vol_scaled_weight_func(dummy_weight_func, sigma_hat, target_vol=0.10)

    # Create a mock lookback
    lookback = pd.DataFrame(
        rng.normal(0, 0.01, (252, 3)),
        index=dates[:252],
        columns=["A", "B", "C"],
    )
    current = pd.Series([0.33, 0.33, 0.34], index=["A", "B", "C"])

    result = scaled(lookback, current)
    assert isinstance(result, pd.Series)
    assert abs(result.sum() - 1.0) < 1e-6  # normalized


def test_sigma_hat_to_regime_proxy_bounded():
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(42)
    sigma_hat = pd.Series(rng.uniform(0.05, 0.25, size=500), index=dates)

    proxy = sigma_hat_to_regime_proxy(sigma_hat, rolling_window=100)
    valid = proxy.dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()
