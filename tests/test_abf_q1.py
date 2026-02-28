"""Tests for harbor.abf.q1 analysis, robustness, and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.abf.q1.analysis import (
    build_control_matrix,
    compute_forward_returns,
    compute_return_autocorrelation,
    fit_local_projection,
    fit_local_projections,
)
from harbor.abf.q1.robustness import apply_shock_definition, robustness_sweep, split_sample
from harbor.risk.regime_detection import detect_vol_shocks, vol_control_pressure_proxy


@pytest.fixture
def synthetic_returns():
    """Generate ~2000 business days of synthetic returns starting 2015-01-05.

    Injects shock clusters at known locations to ensure detectable shocks.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2015-01-05", periods=2000, freq="B")
    # Base: small-vol normal returns
    rets = np.random.normal(0.0003, 0.01, size=len(dates))

    # Inject shock clusters at indices 500-505, 1000-1005, 1500-1505
    for start in [500, 1000, 1500]:
        rets[start : start + 5] = np.random.normal(-0.03, 0.04, size=5)

    return pd.Series(rets, index=dates, name="synthetic")


@pytest.fixture
def shocks_and_proxy(synthetic_returns):
    """Pre-computed shocks and vol proxy from synthetic returns."""
    shocks = detect_vol_shocks(synthetic_returns, threshold_pct=0.95, vol_window=21)
    proxy = vol_control_pressure_proxy(synthetic_returns)
    return shocks, proxy


class TestForwardReturns:
    def test_shape_and_columns(self, synthetic_returns):
        fwd = compute_forward_returns(synthetic_returns, horizons=[1, 5])
        assert fwd.shape[0] == len(synthetic_returns)
        assert "fwd_1" in fwd.columns
        assert "fwd_5" in fwd.columns

    def test_alignment(self, synthetic_returns):
        fwd = compute_forward_returns(synthetic_returns, horizons=[1])
        # Last row should be NaN (no forward data)
        assert pd.isna(fwd["fwd_1"].iloc[-1])

    def test_default_horizons(self, synthetic_returns):
        fwd = compute_forward_returns(synthetic_returns)
        assert set(fwd.columns) == {"fwd_1", "fwd_5", "fwd_21"}


class TestReturnAutocorrelation:
    def test_output_columns(self, synthetic_returns, shocks_and_proxy):
        shocks, _ = shocks_and_proxy
        result = compute_return_autocorrelation(synthetic_returns, shocks)
        assert "autocorr" in result.columns
        assert "regime" in result.columns

    def test_regime_values(self, synthetic_returns, shocks_and_proxy):
        shocks, _ = shocks_and_proxy
        result = compute_return_autocorrelation(synthetic_returns, shocks)
        valid = result["regime"].dropna().unique()
        for v in valid:
            assert v in ("shock", "normal")


class TestControlMatrix:
    def test_columns_no_market(self, synthetic_returns):
        ctrl = build_control_matrix(synthetic_returns)
        assert "baseline_volatility" in ctrl.columns
        assert "liquidity_proxy" in ctrl.columns
        assert "market_beta" not in ctrl.columns

    def test_columns_with_market(self, synthetic_returns):
        market = synthetic_returns * 0.5 + np.random.normal(0, 0.005, len(synthetic_returns))
        market = pd.Series(market, index=synthetic_returns.index)
        ctrl = build_control_matrix(synthetic_returns, market_returns=market)
        assert "market_beta" in ctrl.columns

    def test_dow_dummies(self, synthetic_returns):
        ctrl = build_control_matrix(synthetic_returns)
        dow_cols = [c for c in ctrl.columns if c.startswith("dow_")]
        assert len(dow_cols) >= 1

    def test_nan_handling(self, synthetic_returns):
        ctrl = build_control_matrix(synthetic_returns)
        # First 20 rows should have NaN for baseline_volatility (21d rolling)
        assert ctrl["baseline_volatility"].iloc[:20].isna().any()
        # But there should be valid values later
        assert ctrl["baseline_volatility"].iloc[25:].notna().any()


class TestLocalProjection:
    def test_returns_statsmodels_result(self, synthetic_returns, shocks_and_proxy):
        shocks, proxy = shocks_and_proxy
        ctrl = build_control_matrix(synthetic_returns)
        result = fit_local_projection(synthetic_returns, shocks, proxy, ctrl, horizon=5)
        # Should have shock and shock_x_vol_proxy params
        assert "shock" in result.params.index
        assert "shock_x_vol_proxy" in result.params.index

    def test_correct_param_count(self, synthetic_returns, shocks_and_proxy):
        shocks, proxy = shocks_and_proxy
        ctrl = build_control_matrix(synthetic_returns)
        result = fit_local_projection(synthetic_returns, shocks, proxy, ctrl, horizon=5)
        # const + controls + shock + shock_x_vol_proxy
        n_ctrl = ctrl.shape[1]
        expected = 1 + n_ctrl + 2  # const + controls + shock + interaction
        assert len(result.params) == expected

    def test_multiple_horizons(self, synthetic_returns, shocks_and_proxy):
        shocks, proxy = shocks_and_proxy
        ctrl = build_control_matrix(synthetic_returns)
        results = fit_local_projections(synthetic_returns, shocks, proxy, ctrl, horizons=[1, 5])
        assert set(results.keys()) == {1, 5}
        for _h, res in results.items():
            assert "shock" in res.params.index


class TestApplyShockDefinition:
    def test_primary_config(self, synthetic_returns):
        cfg = {
            "method": "realized_vol_change_percentile",
            "threshold_percentile": 0.95,
            "lookback_days": 21,
        }
        shocks = apply_shock_definition(synthetic_returns, cfg)
        assert shocks.dtype == bool
        assert shocks.sum() > 0

    def test_level_percentile(self, synthetic_returns):
        cfg = {
            "method": "realized_vol_level_percentile",
            "threshold_percentile": 0.95,
            "lookback_days": 63,
        }
        shocks = apply_shock_definition(synthetic_returns, cfg)
        assert shocks.dtype == bool

    def test_range_based(self, synthetic_returns):
        cfg = {
            "method": "range_based_volatility_jump",
            "threshold_percentile": 0.95,
            "lookback_days": 21,
        }
        shocks = apply_shock_definition(synthetic_returns, cfg)
        assert shocks.dtype == bool

    def test_vix_fallback_warning(self, synthetic_returns):
        cfg = {"method": "vix_jump", "threshold_points": 5.0}
        with pytest.warns(UserWarning, match="vix_jump"):
            shocks = apply_shock_definition(synthetic_returns, cfg)
        assert shocks.dtype == bool

    def test_unknown_method_raises(self, synthetic_returns):
        with pytest.raises(ValueError, match="Unknown shock method"):
            apply_shock_definition(synthetic_returns, {"method": "nonexistent"})


class TestSplitSample:
    def test_full(self, synthetic_returns):
        mask = split_sample(synthetic_returns, "full")
        assert mask.all()

    def test_pre_post_coverage(self, synthetic_returns):
        pre = split_sample(synthetic_returns, "pre_2020")
        post = split_sample(synthetic_returns, "post_2020")
        # pre + post should cover the full index
        assert (pre | post).all()
        # No overlap
        assert not (pre & post).any()

    def test_liquidity_splits(self, synthetic_returns):
        hi = split_sample(synthetic_returns, "high_liquidity")
        lo = split_sample(synthetic_returns, "low_liquidity")
        # Together they cover all non-NaN observations
        combined = hi | lo
        # Some early NaN rows from rolling might both be False
        assert combined.sum() >= len(synthetic_returns) - 25

    def test_unknown_split_raises(self, synthetic_returns):
        with pytest.raises(ValueError, match="Unknown sample split"):
            split_sample(synthetic_returns, "nonexistent")


class TestRobustnessSweep:
    def test_returns_expected_schema(self, synthetic_returns, shocks_and_proxy):
        _, proxy = shocks_and_proxy
        ctrl = build_control_matrix(synthetic_returns)

        configs = [
            {
                "method": "realized_vol_change_percentile",
                "threshold_percentile": 0.95,
                "lookback_days": 21,
            }
        ]
        splits = ["full"]
        horizons = [1, 5]

        result = robustness_sweep(synthetic_returns, configs, splits, horizons, proxy, ctrl)

        expected_cols = {
            "shock_method",
            "sample_split",
            "horizon",
            "b_h",
            "b_h_se",
            "b_h_pval",
            "c_h",
            "c_h_se",
            "c_h_pval",
            "r_squared",
            "n_obs",
        }
        assert expected_cols == set(result.columns)
        assert len(result) == 2  # 1 config * 1 split * 2 horizons

    def test_multiple_combos(self, synthetic_returns, shocks_and_proxy):
        _, proxy = shocks_and_proxy
        ctrl = build_control_matrix(synthetic_returns)

        configs = [
            {
                "method": "realized_vol_change_percentile",
                "threshold_percentile": 0.95,
                "lookback_days": 21,
            },
            {
                "method": "realized_vol_level_percentile",
                "threshold_percentile": 0.95,
                "lookback_days": 63,
            },
        ]
        splits = ["full", "pre_2020"]
        horizons = [1, 5]

        result = robustness_sweep(synthetic_returns, configs, splits, horizons, proxy, ctrl)
        assert len(result) == 2 * 2 * 2  # 2 configs * 2 splits * 2 horizons
