"""Tests for hangar.risk — covariance, HRP, Monte Carlo, regime detection, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_multi_returns() -> pd.DataFrame:
    """500 business-day return panel for 4 synthetic assets."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    data = rng.normal(0.0003, 0.01, size=(500, 4))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


@pytest.fixture
def synthetic_series_returns(synthetic_multi_returns) -> pd.Series:
    """Single-asset return series."""
    return synthetic_multi_returns["A"]


@pytest.fixture
def equal_weights() -> pd.Series:
    return pd.Series(0.25, index=["A", "B", "C", "D"])


# ===================================================================
# hangar.risk.covariance
# ===================================================================

class TestSampleCovariance:
    def test_shape_and_symmetry(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        cov = sample_covariance(synthetic_multi_returns)
        assert cov.shape == (4, 4)
        pd.testing.assert_frame_equal(cov, cov.T)

    def test_positive_diagonal(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        cov = sample_covariance(synthetic_multi_returns)
        assert (np.diag(cov.values) > 0).all()

    def test_annualization_scaling(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        cov1 = sample_covariance(synthetic_multi_returns, annualization=1)
        cov252 = sample_covariance(synthetic_multi_returns, annualization=252)
        np.testing.assert_allclose(cov252.values, cov1.values * 252, rtol=1e-8)


class TestShrinkageCovariance:
    def test_ledoit_wolf_shape(self, synthetic_multi_returns):
        from hangar.risk.covariance import shrinkage_covariance
        cov = shrinkage_covariance(synthetic_multi_returns, method="ledoit_wolf")
        assert cov.shape == (4, 4)

    def test_oas_shape(self, synthetic_multi_returns):
        from hangar.risk.covariance import shrinkage_covariance
        cov = shrinkage_covariance(synthetic_multi_returns, method="oas")
        assert cov.shape == (4, 4)

    def test_symmetry(self, synthetic_multi_returns):
        from hangar.risk.covariance import shrinkage_covariance
        cov = shrinkage_covariance(synthetic_multi_returns)
        pd.testing.assert_frame_equal(cov, cov.T)


class TestEstimateCovariance:
    def test_sample_dispatch(self, synthetic_multi_returns):
        from hangar.risk.covariance import estimate_covariance, sample_covariance
        cov_est = estimate_covariance(synthetic_multi_returns, method="sample")
        cov_direct = sample_covariance(synthetic_multi_returns)
        pd.testing.assert_frame_equal(cov_est, cov_direct)

    def test_ledoit_wolf_dispatch(self, synthetic_multi_returns):
        from hangar.risk.covariance import estimate_covariance
        cov = estimate_covariance(synthetic_multi_returns, method="ledoit_wolf")
        assert cov.shape == (4, 4)

    def test_unknown_method_raises(self, synthetic_multi_returns):
        from hangar.risk.covariance import estimate_covariance
        with pytest.raises(ValueError, match="Unknown covariance method"):
            estimate_covariance(synthetic_multi_returns, method="magic")


class TestCovarianceValidation:
    def test_non_dataframe_raises(self):
        from hangar.risk.covariance import sample_covariance
        with pytest.raises(TypeError, match="pandas DataFrame"):
            sample_covariance(np.array([[1, 2], [3, 4]]))

    def test_empty_dataframe_raises(self):
        from hangar.risk.covariance import sample_covariance
        with pytest.raises(ValueError, match="empty"):
            sample_covariance(pd.DataFrame())

    def test_single_row_raises(self):
        from hangar.risk.covariance import sample_covariance
        df = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValueError, match="at least two rows"):
            sample_covariance(df)

    def test_single_asset_raises(self):
        from hangar.risk.covariance import sample_covariance
        df = pd.DataFrame({"A": [0.01, 0.02, 0.03]})
        with pytest.raises(ValueError, match="at least two assets"):
            sample_covariance(df)


# ===================================================================
# hangar.risk.hrp
# ===================================================================

class TestHrpAllocation:
    def test_weights_sum_to_one(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import hrp_allocation
        cov = sample_covariance(synthetic_multi_returns)
        weights = hrp_allocation(cov)
        assert abs(weights.sum() - 1.0) < 1e-8

    def test_all_weights_positive(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import hrp_allocation
        cov = sample_covariance(synthetic_multi_returns)
        weights = hrp_allocation(cov)
        assert (weights > 0).all()

    def test_weights_index_matches_cov(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import hrp_allocation
        cov = sample_covariance(synthetic_multi_returns)
        weights = hrp_allocation(cov)
        assert set(weights.index) == set(cov.index)

    def test_linkage_methods(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import hrp_allocation
        cov = sample_covariance(synthetic_multi_returns)
        for method in ("single", "complete", "average"):
            w = hrp_allocation(cov, linkage_method=method)
            assert abs(w.sum() - 1.0) < 1e-8

    def test_two_asset_case(self):
        from hangar.risk.hrp import hrp_allocation
        cov = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]], index=["X", "Y"], columns=["X", "Y"]
        )
        w = hrp_allocation(cov)
        assert abs(w.sum() - 1.0) < 1e-8
        # Lower-variance asset X should get higher weight
        assert w["X"] > w["Y"]


class TestCovToCorr:
    def test_diagonal_is_one(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import cov_to_corr
        cov = sample_covariance(synthetic_multi_returns)
        corr = cov_to_corr(cov)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_bounded(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.hrp import cov_to_corr
        cov = sample_covariance(synthetic_multi_returns)
        corr = cov_to_corr(cov)
        assert (corr.values >= -1.0).all()
        assert (corr.values <= 1.0).all()


class TestHrpValidation:
    def test_non_dataframe_raises(self):
        from hangar.risk.hrp import hrp_allocation
        with pytest.raises(TypeError, match="pandas DataFrame"):
            hrp_allocation(np.eye(3))

    def test_empty_raises(self):
        from hangar.risk.hrp import hrp_allocation
        with pytest.raises(ValueError, match="empty"):
            hrp_allocation(pd.DataFrame())

    def test_non_square_raises(self):
        from hangar.risk.hrp import hrp_allocation
        df = pd.DataFrame(np.ones((2, 3)), columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="square"):
            hrp_allocation(df)

    def test_mismatched_labels_raises(self):
        from hangar.risk.hrp import hrp_allocation
        df = pd.DataFrame(
            np.eye(2), index=["A", "B"], columns=["X", "Y"]
        )
        with pytest.raises(ValueError, match="index and columns must match"):
            hrp_allocation(df)

    def test_nan_raises(self):
        from hangar.risk.hrp import hrp_allocation
        df = pd.DataFrame(
            [[1.0, np.nan], [np.nan, 1.0]], index=["A", "B"], columns=["A", "B"]
        )
        with pytest.raises(ValueError, match="NaN"):
            hrp_allocation(df)


# ===================================================================
# hangar.risk.monte_carlo
# ===================================================================

class TestSimulateMultivariateReturns:
    def test_output_shape(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.monte_carlo import simulate_multivariate_returns
        mean = synthetic_multi_returns.mean()
        cov = sample_covariance(synthetic_multi_returns, annualization=1)
        sims = simulate_multivariate_returns(mean, cov, n_sims=100, horizon=5, random_state=0)
        assert sims.shape == (100, 5, 4)

    def test_deterministic_with_seed(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.monte_carlo import simulate_multivariate_returns
        mean = synthetic_multi_returns.mean()
        cov = sample_covariance(synthetic_multi_returns, annualization=1)
        s1 = simulate_multivariate_returns(mean, cov, n_sims=50, horizon=3, random_state=99)
        s2 = simulate_multivariate_returns(mean, cov, n_sims=50, horizon=3, random_state=99)
        np.testing.assert_array_equal(s1, s2)

    def test_invalid_n_sims_raises(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.monte_carlo import simulate_multivariate_returns
        mean = synthetic_multi_returns.mean()
        cov = sample_covariance(synthetic_multi_returns, annualization=1)
        with pytest.raises(ValueError, match="n_sims"):
            simulate_multivariate_returns(mean, cov, n_sims=0)

    def test_invalid_horizon_raises(self, synthetic_multi_returns):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.monte_carlo import simulate_multivariate_returns
        mean = synthetic_multi_returns.mean()
        cov = sample_covariance(synthetic_multi_returns, annualization=1)
        with pytest.raises(ValueError, match="horizon"):
            simulate_multivariate_returns(mean, cov, horizon=0)


class TestPortfolioVarCvar:
    def test_basic_result(self, synthetic_multi_returns, equal_weights):
        from hangar.risk.covariance import sample_covariance
        from hangar.risk.monte_carlo import portfolio_var_cvar, simulate_multivariate_returns
        mean = synthetic_multi_returns.mean()
        cov = sample_covariance(synthetic_multi_returns, annualization=1)
        sims = simulate_multivariate_returns(mean, cov, n_sims=5000, horizon=21, random_state=42)
        result = portfolio_var_cvar(equal_weights, sims, alpha=0.95)
        assert result.alpha == 0.95
        assert result.cvar >= result.var  # CVaR dominates VaR
        assert np.isfinite(result.expected_return)

    def test_invalid_alpha_raises(self, equal_weights):
        from hangar.risk.monte_carlo import portfolio_var_cvar
        sims = np.random.default_rng(0).normal(size=(10, 5, 4))
        with pytest.raises(ValueError, match="alpha"):
            portfolio_var_cvar(equal_weights, sims, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            portfolio_var_cvar(equal_weights, sims, alpha=1.0)

    def test_wrong_dimensions_raises(self, equal_weights):
        from hangar.risk.monte_carlo import portfolio_var_cvar
        sims_2d = np.random.default_rng(0).normal(size=(10, 4))
        with pytest.raises(ValueError, match="3D"):
            portfolio_var_cvar(equal_weights, sims_2d)

    def test_weight_dimension_mismatch_raises(self):
        from hangar.risk.monte_carlo import portfolio_var_cvar
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        sims = np.random.default_rng(0).normal(size=(10, 5, 4))
        with pytest.raises(ValueError, match="weights length"):
            portfolio_var_cvar(w, sims)


class TestMonteCarloFromHistory:
    def test_end_to_end(self, synthetic_multi_returns, equal_weights):
        from hangar.risk.monte_carlo import monte_carlo_var_cvar_from_history
        result = monte_carlo_var_cvar_from_history(
            synthetic_multi_returns, equal_weights,
            n_sims=1000, horizon=5, random_state=42,
        )
        assert result.alpha == 0.95
        assert result.cvar >= result.var
        assert np.isfinite(result.var)
        assert np.isfinite(result.expected_return)

    def test_empty_aligned_raises(self, equal_weights):
        from hangar.risk.monte_carlo import monte_carlo_var_cvar_from_history
        df = pd.DataFrame(
            {"A": [np.nan], "B": [np.nan], "C": [np.nan], "D": [np.nan]},
            index=pd.bdate_range("2020-01-01", periods=1),
        )
        with pytest.raises(ValueError, match="No aligned return history"):
            monte_carlo_var_cvar_from_history(df, equal_weights)


# ===================================================================
# hangar.risk.regime_detection
# ===================================================================

class TestDetectVolShocks:
    def test_returns_bool_series(self, synthetic_series_returns):
        from hangar.risk.regime_detection import detect_vol_shocks
        shocks = detect_vol_shocks(synthetic_series_returns)
        assert shocks.dtype == bool
        assert shocks.name == "vol_shock"

    def test_accepts_dataframe(self, synthetic_multi_returns):
        from hangar.risk.regime_detection import detect_vol_shocks
        shocks = detect_vol_shocks(synthetic_multi_returns)
        assert shocks.dtype == bool

    def test_threshold_controls_rate(self, synthetic_series_returns):
        from hangar.risk.regime_detection import detect_vol_shocks
        conservative = detect_vol_shocks(synthetic_series_returns, threshold_pct=0.99)
        aggressive = detect_vol_shocks(synthetic_series_returns, threshold_pct=0.50)
        assert conservative.sum() <= aggressive.sum()

    def test_invalid_threshold_raises(self, synthetic_series_returns):
        from hangar.risk.regime_detection import detect_vol_shocks
        with pytest.raises(ValueError, match="threshold_pct"):
            detect_vol_shocks(synthetic_series_returns, threshold_pct=0.0)
        with pytest.raises(ValueError, match="threshold_pct"):
            detect_vol_shocks(synthetic_series_returns, threshold_pct=1.0)

    def test_small_vol_window_raises(self, synthetic_series_returns):
        from hangar.risk.regime_detection import detect_vol_shocks
        with pytest.raises(ValueError, match="vol_window"):
            detect_vol_shocks(synthetic_series_returns, vol_window=1)


class TestVolControlPressureProxy:
    def test_returns_named_series(self, synthetic_series_returns):
        from hangar.risk.regime_detection import vol_control_pressure_proxy
        proxy = vol_control_pressure_proxy(synthetic_series_returns)
        assert proxy.name == "vol_control_pressure"
        assert isinstance(proxy, pd.Series)

    def test_bounded_0_2(self, synthetic_series_returns):
        from hangar.risk.regime_detection import vol_control_pressure_proxy
        proxy = vol_control_pressure_proxy(synthetic_series_returns)
        valid = proxy.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 2.0).all()

    def test_accepts_dataframe(self, synthetic_multi_returns):
        from hangar.risk.regime_detection import vol_control_pressure_proxy
        proxy = vol_control_pressure_proxy(synthetic_multi_returns)
        assert isinstance(proxy, pd.Series)

    def test_bad_window_raises(self, synthetic_series_returns):
        from hangar.risk.regime_detection import vol_control_pressure_proxy
        with pytest.raises(ValueError, match="Require"):
            vol_control_pressure_proxy(synthetic_series_returns, short_window=1)
        with pytest.raises(ValueError, match="Require"):
            vol_control_pressure_proxy(
                synthetic_series_returns, short_window=100, long_window=50
            )


class TestCoerceMarketReturns:
    def test_empty_dataframe_raises(self):
        from hangar.risk.regime_detection import _coerce_market_returns
        with pytest.raises(ValueError, match="empty"):
            _coerce_market_returns(pd.DataFrame())

    def test_wrong_type_raises(self):
        from hangar.risk.regime_detection import _coerce_market_returns
        with pytest.raises(TypeError, match="pandas"):
            _coerce_market_returns([0.01, 0.02])

    def test_all_nan_raises(self):
        from hangar.risk.regime_detection import _coerce_market_returns
        s = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="no valid observations"):
            _coerce_market_returns(s)


# ===================================================================
# hangar.risk.correlation (stub — expected to raise)
# ===================================================================

class TestCorrelationStub:
    def test_detect_correlation_spikes_not_implemented(self, synthetic_multi_returns):
        from hangar.risk.correlation import detect_correlation_spikes
        with pytest.raises(NotImplementedError, match="Layer 4 roadmap"):
            detect_correlation_spikes(synthetic_multi_returns)


# ===================================================================
# hangar.risk.__init__ re-exports
# ===================================================================

class TestRiskPublicApi:
    def test_all_exports_importable(self):
        import hangar.risk as risk_mod
        from hangar.risk import __all__
        for name in __all__:
            assert hasattr(risk_mod, name), f"{name} listed in __all__ but not importable"
