"""Tests for classical volatility forecasting baselines (GARCH, EWMA, rolling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.ml.volatility.baselines import (
    evaluate_forecast,
    ewma_volatility,
    fit_garch11,
    garch11_forecast,
    rolling_volatility,
    run_baseline_comparison,
)


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """Generate synthetic daily returns with known volatility structure."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Two-regime returns: low vol then high vol
    r1 = rng.normal(0, 0.01, n // 2)
    r2 = rng.normal(0, 0.03, n - n // 2)
    returns = np.concatenate([r1, r2])
    return pd.Series(returns, index=dates, name="returns")


class TestEWMA:
    def test_output_shape(self, synthetic_returns):
        vol = ewma_volatility(synthetic_returns)
        assert len(vol) == len(synthetic_returns)
        assert vol.name == "ewma_vol"

    def test_non_negative(self, synthetic_returns):
        vol = ewma_volatility(synthetic_returns).dropna()
        assert (vol >= 0).all()

    def test_higher_vol_in_high_regime(self, synthetic_returns):
        vol = ewma_volatility(synthetic_returns).dropna()
        mid = len(vol) // 2
        # Second half should have higher vol (after burn-in)
        late_vol = vol.iloc[mid + 50 :].mean()
        early_vol = vol.iloc[50:mid].mean()
        assert late_vol > early_vol

    def test_horizon_scaling(self, synthetic_returns):
        vol1 = ewma_volatility(synthetic_returns, horizon=1).dropna()
        vol5 = ewma_volatility(synthetic_returns, horizon=5).dropna()
        # 5-day vol should be ~sqrt(5) times 1-day vol
        ratio = (vol5 / vol1).dropna().median()
        assert 2.0 < ratio < 2.5  # sqrt(5) ≈ 2.24


class TestRollingVol:
    def test_output_shape(self, synthetic_returns):
        vol = rolling_volatility(synthetic_returns, window=21)
        assert len(vol) == len(synthetic_returns)

    def test_nan_for_initial_window(self, synthetic_returns):
        vol = rolling_volatility(synthetic_returns, window=21)
        assert vol.iloc[:20].isna().all()
        assert vol.iloc[21:].notna().all()

    def test_annualization(self, synthetic_returns):
        vol = rolling_volatility(synthetic_returns, window=21).dropna()
        vol_ann = rolling_volatility(synthetic_returns, window=21, annualize=True).dropna()
        ratio = (vol_ann / vol).dropna().median()
        np.testing.assert_allclose(ratio, np.sqrt(252), rtol=0.01)


class TestGARCH:
    def test_fit_converges(self, synthetic_returns):
        params = fit_garch11(synthetic_returns)
        assert params["converged"]
        assert 0 < params["alpha"] < 1
        assert 0 < params["beta"] < 1
        assert params["alpha"] + params["beta"] < 1

    def test_persistence_reasonable(self, synthetic_returns):
        params = fit_garch11(synthetic_returns)
        # Typical GARCH persistence is 0.85-0.99
        assert 0.5 < params["persistence"] < 1.0

    def test_forecast_shape(self, synthetic_returns):
        vol = garch11_forecast(synthetic_returns)
        assert len(vol) == len(synthetic_returns.dropna())
        assert vol.name == "garch_vol"

    def test_forecast_positive(self, synthetic_returns):
        vol = garch11_forecast(synthetic_returns)
        assert (vol > 0).all()

    def test_forecast_with_given_params(self, synthetic_returns):
        params = fit_garch11(synthetic_returns)
        vol = garch11_forecast(
            synthetic_returns,
            omega=params["omega"],
            alpha=params["alpha"],
            beta=params["beta"],
        )
        assert len(vol) == len(synthetic_returns.dropna())


class TestEvaluateForecast:
    def test_perfect_forecast(self):
        actual = pd.Series([0.01, 0.02, 0.03, 0.015, 0.025])
        metrics = evaluate_forecast(actual, actual)
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_metrics_keys(self, synthetic_returns):
        vol = ewma_volatility(synthetic_returns)
        actual = synthetic_returns.abs()
        metrics = evaluate_forecast(vol, actual)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "qlike" in metrics
        assert "directional_accuracy" in metrics

    def test_directional_accuracy_range(self, synthetic_returns):
        vol = ewma_volatility(synthetic_returns)
        actual = synthetic_returns.abs()
        metrics = evaluate_forecast(vol, actual)
        assert 0 <= metrics["directional_accuracy"] <= 1


class TestBaselineComparison:
    def test_runs_all_models(self, synthetic_returns):
        df = run_baseline_comparison(synthetic_returns)
        models = df.index.get_level_values("model").unique()
        assert "EWMA" in models
        assert "Rolling_21d" in models
        assert "GARCH(1,1)" in models

    def test_output_columns(self, synthetic_returns):
        df = run_baseline_comparison(synthetic_returns)
        assert "rmse" in df.columns
        assert "mae" in df.columns
        assert "qlike" in df.columns
        assert "directional_accuracy" in df.columns

    def test_all_metrics_finite(self, synthetic_returns):
        df = run_baseline_comparison(synthetic_returns)
        assert df[["rmse", "mae"]].notna().all().all()
