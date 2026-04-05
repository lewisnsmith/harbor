"""Tests for HANGAR H2 risk modules — regime covariance, non-Gaussian MC,
scenarios, decomposition, and risk engine."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """500 business-day return panel for 4 synthetic assets."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    data = rng.normal(0.0003, 0.01, size=(500, 4))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


@pytest.fixture
def equal_weights() -> pd.Series:
    return pd.Series(0.25, index=["A", "B", "C", "D"])


@pytest.fixture
def cov_matrix(synthetic_returns) -> pd.DataFrame:
    from hangar.risk.covariance import sample_covariance
    return sample_covariance(synthetic_returns, annualization=1)


@pytest.fixture
def mean_returns(synthetic_returns) -> pd.Series:
    return synthetic_returns.mean()


# ===================================================================
# hangar.risk.covariance — H2 regime-aware extensions
# ===================================================================

class TestRegimeAwareCovariance:
    def test_returns_dict_of_dataframes(self, synthetic_returns):
        from hangar.risk.covariance import regime_aware_covariance
        labels = pd.Series(
            ["A"] * 250 + ["B"] * 250, index=synthetic_returns.index
        )
        result = regime_aware_covariance(synthetic_returns, labels, method="sample")
        assert isinstance(result, dict)
        assert "A" in result and "B" in result
        assert result["A"].shape == (4, 4)

    def test_single_regime(self, synthetic_returns):
        from hangar.risk.covariance import regime_aware_covariance
        labels = pd.Series("only", index=synthetic_returns.index)
        result = regime_aware_covariance(synthetic_returns, labels, method="sample")
        assert "only" in result
        assert len(result) == 1

    def test_no_overlap_raises(self, synthetic_returns):
        from hangar.risk.covariance import regime_aware_covariance
        bad_index = pd.date_range("2099-01-01", periods=10, freq="B")
        labels = pd.Series("x", index=bad_index)
        with pytest.raises(ValueError, match="no overlap"):
            regime_aware_covariance(synthetic_returns, labels)

    def test_regime_with_one_obs_skipped(self, synthetic_returns):
        from hangar.risk.covariance import regime_aware_covariance
        labels = pd.Series("big", index=synthetic_returns.index)
        labels.iloc[0] = "tiny"  # only 1 observation
        result = regime_aware_covariance(synthetic_returns, labels, method="sample")
        assert "tiny" not in result
        assert "big" in result


class TestExpandingRegimeCovariance:
    def test_returns_high_low_keys(self, synthetic_returns):
        from hangar.risk.covariance import expanding_regime_covariance
        result = expanding_regime_covariance(synthetic_returns, vol_threshold_pct=0.8)
        assert isinstance(result, dict)
        assert any(k in result for k in ["high_vol", "low_vol"])

    def test_high_vol_matrix_shape(self, synthetic_returns):
        from hangar.risk.covariance import expanding_regime_covariance
        result = expanding_regime_covariance(synthetic_returns, vol_threshold_pct=0.5)
        for cov in result.values():
            assert cov.shape == (4, 4)


class TestEstimateCovarianceRegimeAware:
    def test_regime_aware_dispatch(self, synthetic_returns):
        from hangar.risk.covariance import estimate_covariance
        cov = estimate_covariance(synthetic_returns, method="regime_aware")
        assert cov.shape == (4, 4)
        assert (np.diag(cov.values) > 0).all()


# ===================================================================
# hangar.risk.monte_carlo — H2 non-Gaussian extensions
# ===================================================================

class TestStudentTSimulation:
    def test_output_shape(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_student_t_returns
        sims = simulate_student_t_returns(
            mean_returns, cov_matrix, df=5, n_sims=100, horizon=5, random_state=0
        )
        assert sims.shape == (100, 5, 4)

    def test_deterministic_with_seed(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_student_t_returns
        s1 = simulate_student_t_returns(mean_returns, cov_matrix, n_sims=50, horizon=3, random_state=99)
        s2 = simulate_student_t_returns(mean_returns, cov_matrix, n_sims=50, horizon=3, random_state=99)
        np.testing.assert_array_equal(s1, s2)

    def test_heavier_tails_than_normal(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_multivariate_returns, simulate_student_t_returns
        normal = simulate_multivariate_returns(mean_returns, cov_matrix, n_sims=50_000, horizon=1, random_state=0)
        student = simulate_student_t_returns(mean_returns, cov_matrix, df=3, n_sims=50_000, horizon=1, random_state=0)
        # Student-t should have higher kurtosis (heavier tails)
        normal_kurt = float(np.mean(normal[:, 0, 0] ** 4) / np.mean(normal[:, 0, 0] ** 2) ** 2)
        student_kurt = float(np.mean(student[:, 0, 0] ** 4) / np.mean(student[:, 0, 0] ** 2) ** 2)
        assert student_kurt > normal_kurt

    def test_invalid_df_raises(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_student_t_returns
        with pytest.raises(ValueError, match="df"):
            simulate_student_t_returns(mean_returns, cov_matrix, df=2)


class TestFactorSimulation:
    def test_output_shape(self):
        from hangar.risk.monte_carlo import simulate_factor_returns
        loadings = pd.DataFrame(
            [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]],
            index=["A", "B", "C"],
            columns=["F1", "F2"],
        )
        f_cov = pd.DataFrame(
            [[0.01, 0.002], [0.002, 0.01]],
            index=["F1", "F2"],
            columns=["F1", "F2"],
        )
        idio = pd.Series([0.001, 0.001, 0.001], index=["A", "B", "C"])
        sims = simulate_factor_returns(loadings, f_cov, idio, n_sims=100, horizon=5, random_state=0)
        assert sims.shape == (100, 5, 3)

    def test_deterministic(self):
        from hangar.risk.monte_carlo import simulate_factor_returns
        loadings = pd.DataFrame([[1.0]], index=["A"], columns=["F1"])
        f_cov = pd.DataFrame([[0.01]], index=["F1"], columns=["F1"])
        idio = pd.Series([0.001], index=["A"])
        s1 = simulate_factor_returns(loadings, f_cov, idio, n_sims=50, horizon=3, random_state=42)
        s2 = simulate_factor_returns(loadings, f_cov, idio, n_sims=50, horizon=3, random_state=42)
        np.testing.assert_array_equal(s1, s2)


class TestSimulateReturnsDispatch:
    def test_normal_dispatch(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_returns
        sims = simulate_returns(mean_returns, cov_matrix, method="normal", n_sims=100, horizon=5, random_state=0)
        assert sims.shape == (100, 5, 4)

    def test_student_t_dispatch(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_returns
        sims = simulate_returns(mean_returns, cov_matrix, method="student_t", n_sims=100, horizon=5, random_state=0, df=5)
        assert sims.shape == (100, 5, 4)

    def test_factor_dispatch_raises(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_returns
        with pytest.raises(ValueError, match="simulate_factor_returns"):
            simulate_returns(mean_returns, cov_matrix, method="factor")

    def test_unknown_method_raises(self, mean_returns, cov_matrix):
        from hangar.risk.monte_carlo import simulate_returns
        with pytest.raises(ValueError, match="Unknown"):
            simulate_returns(mean_returns, cov_matrix, method="magic")


class TestMonteCarloFromHistoryH2:
    def test_student_t_method(self, synthetic_returns, equal_weights):
        from hangar.risk.monte_carlo import monte_carlo_var_cvar_from_history
        result = monte_carlo_var_cvar_from_history(
            synthetic_returns, equal_weights,
            n_sims=1000, horizon=5, random_state=42,
            simulation_method="student_t", simulation_kwargs={"df": 5},
        )
        assert result.cvar >= result.var
        assert np.isfinite(result.var)


# ===================================================================
# hangar.risk.scenarios
# ===================================================================

class TestApplyVolSpike:
    def test_variances_scaled(self, cov_matrix):
        from hangar.risk.scenarios import apply_vol_spike
        stressed = apply_vol_spike(cov_matrix, multiplier=4.0)
        # Diagonal (variances) should be ~4x
        np.testing.assert_allclose(
            np.diag(stressed.values), np.diag(cov_matrix.values) * 4.0, rtol=1e-6
        )

    def test_correlations_preserved(self, cov_matrix):
        from hangar.risk.scenarios import apply_vol_spike
        stressed = apply_vol_spike(cov_matrix, multiplier=2.0)
        # Extract correlation matrices
        stds_orig = np.sqrt(np.diag(cov_matrix.values))
        corr_orig = cov_matrix.values / np.outer(stds_orig, stds_orig)
        stds_new = np.sqrt(np.diag(stressed.values))
        corr_new = stressed.values / np.outer(stds_new, stds_new)
        np.testing.assert_allclose(corr_orig, corr_new, atol=1e-6)

    def test_invalid_multiplier_raises(self, cov_matrix):
        from hangar.risk.scenarios import apply_vol_spike
        with pytest.raises(ValueError, match="multiplier"):
            apply_vol_spike(cov_matrix, multiplier=0.0)


class TestApplyCorrelationSpike:
    def test_off_diag_set(self, cov_matrix):
        from hangar.risk.scenarios import apply_correlation_spike
        stressed = apply_correlation_spike(cov_matrix, target_corr=0.9)
        stds = np.sqrt(np.diag(stressed.values))
        corr = stressed.values / np.outer(stds, stds)
        n = len(stds)
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert abs(corr[i, j] - 1.0) < 1e-6
                else:
                    assert abs(corr[i, j] - 0.9) < 1e-6

    def test_variances_unchanged(self, cov_matrix):
        from hangar.risk.scenarios import apply_correlation_spike
        stressed = apply_correlation_spike(cov_matrix, target_corr=0.5)
        np.testing.assert_allclose(
            np.diag(stressed.values), np.diag(cov_matrix.values), rtol=1e-6
        )


class TestApplySectorCrash:
    def test_sector_shocked(self, mean_returns):
        from hangar.risk.scenarios import apply_sector_crash
        sector_map = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
        shocked = apply_sector_crash(mean_returns, sector_map, "Tech", crash_magnitude=-0.10)
        assert shocked["A"] == pytest.approx(mean_returns["A"] - 0.10)
        assert shocked["B"] == pytest.approx(mean_returns["B"] - 0.10)
        assert shocked["C"] == pytest.approx(mean_returns["C"])
        assert shocked["D"] == pytest.approx(mean_returns["D"])

    def test_unknown_sector_raises(self, mean_returns):
        from hangar.risk.scenarios import apply_sector_crash
        with pytest.raises(ValueError, match="not found"):
            apply_sector_crash(mean_returns, {"A": "Tech"}, "Finance")


class TestRunScenarioSuite:
    def test_runs_multiple_scenarios(self, equal_weights, mean_returns, cov_matrix):
        from hangar.risk.scenarios import run_scenario_suite
        config = [
            {"name": "vol2x", "type": "vol_spike", "params": {"multiplier": 2.0}},
            {"name": "corr90", "type": "correlation_spike", "params": {"target_corr": 0.90}},
        ]
        results = run_scenario_suite(
            equal_weights, mean_returns, cov_matrix, config,
            n_sims=500, horizon=5, random_state=42,
        )
        assert len(results) == 2
        assert results[0].name == "vol2x"
        assert results[1].name == "corr90"
        # Vol spike should increase VaR
        assert results[0].stressed_var >= results[0].baseline_var

    def test_sector_crash_requires_map(self, equal_weights, mean_returns, cov_matrix):
        from hangar.risk.scenarios import run_scenario_suite
        config = [{"name": "crash", "type": "sector_crash", "params": {"crash_sector": "Tech"}}]
        with pytest.raises(ValueError, match="sector_map"):
            run_scenario_suite(equal_weights, mean_returns, cov_matrix, config)


class TestScenarioReport:
    def test_serializable(self, equal_weights, mean_returns, cov_matrix):
        from hangar.risk.scenarios import run_scenario_suite, scenario_report_to_dict
        config = [{"name": "vol2x", "type": "vol_spike", "params": {"multiplier": 2.0}}]
        results = run_scenario_suite(
            equal_weights, mean_returns, cov_matrix, config,
            n_sims=500, horizon=5, random_state=42,
        )
        report = scenario_report_to_dict(results)
        # Should be JSON-serializable
        json_str = json.dumps(report)
        assert "vol2x" in json_str


# ===================================================================
# hangar.risk.decomposition
# ===================================================================

class TestMarginalContributionToRisk:
    def test_shape(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import marginal_contribution_to_risk
        mcr = marginal_contribution_to_risk(equal_weights, cov_matrix)
        assert len(mcr) == 4
        assert mcr.name == "mcr"

    def test_non_negative_for_positive_weights(self, cov_matrix):
        from hangar.risk.decomposition import marginal_contribution_to_risk
        w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov_matrix.index)
        mcr = marginal_contribution_to_risk(w, cov_matrix)
        assert (mcr >= 0).all()


class TestComponentRisk:
    def test_sums_to_portfolio_vol(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import component_risk
        cr = component_risk(equal_weights, cov_matrix)
        w = equal_weights.to_numpy(dtype=float)
        cov = cov_matrix.to_numpy(dtype=float)
        port_vol = np.sqrt(w @ cov @ w)
        assert abs(cr.sum() - port_vol) < 1e-8


class TestPercentRiskContribution:
    def test_sums_to_one(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import percent_risk_contribution
        prc = percent_risk_contribution(equal_weights, cov_matrix)
        assert abs(prc.sum() - 1.0) < 1e-8


class TestFactorRiskDecomposition:
    def test_variance_adds_up(self):
        from hangar.risk.decomposition import factor_risk_decomposition
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        loadings = pd.DataFrame([[0.8, 0.2], [0.3, 0.7]], index=["A", "B"], columns=["F1", "F2"])
        f_cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.04]], index=["F1", "F2"], columns=["F1", "F2"])
        idio = pd.Series([0.001, 0.002], index=["A", "B"])
        result = factor_risk_decomposition(w, loadings, f_cov, idio)
        assert abs(result["total_variance"] - (result["systematic_variance"] + result["idiosyncratic_variance"])) < 1e-10
        assert 0.0 <= result["systematic_pct"] <= 1.0

    def test_factor_contributions_sum(self):
        from hangar.risk.decomposition import factor_risk_decomposition
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        loadings = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["A", "B"], columns=["F1", "F2"])
        f_cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["F1", "F2"], columns=["F1", "F2"])
        idio = pd.Series([0.0, 0.0], index=["A", "B"])
        result = factor_risk_decomposition(w, loadings, f_cov, idio)
        assert abs(result["factor_contributions"].sum() - result["systematic_variance"]) < 1e-10


class TestClusterRiskAttribution:
    def test_output_columns(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import cluster_risk_attribution
        cluster_map = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
        df = cluster_risk_attribution(equal_weights, cov_matrix, cluster_map)
        assert set(df.columns) == {"cluster", "weight", "risk_contribution", "risk_pct"}
        assert len(df) == 2

    def test_risk_pct_sums_to_one(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import cluster_risk_attribution
        cluster_map = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
        df = cluster_risk_attribution(equal_weights, cov_matrix, cluster_map)
        assert abs(df["risk_pct"].sum() - 1.0) < 1e-6


class TestConcentrationMetrics:
    def test_equal_weights_effective_n(self, equal_weights, cov_matrix):
        from hangar.risk.decomposition import concentration_metrics
        metrics = concentration_metrics(equal_weights, cov_matrix)
        # Equal weights -> HHI = 1/4, effective_n = 4
        assert abs(metrics["herfindahl_weight"] - 0.25) < 1e-8
        assert abs(metrics["effective_n_weight"] - 4.0) < 1e-8
        assert metrics["max_risk_contributor"] in ["A", "B", "C", "D"]


# ===================================================================
# hangar.risk.engine
# ===================================================================

class TestRiskEngine:
    def test_default_config(self):
        from hangar.risk.engine import RiskConfig, RiskEngine
        engine = RiskEngine()
        assert engine.config.covariance_method == "ledoit_wolf"
        assert engine.config.simulation_method == "normal"

    def test_estimate_covariance(self, synthetic_returns):
        from hangar.risk.engine import RiskEngine
        engine = RiskEngine()
        cov = engine.estimate_covariance(synthetic_returns)
        assert cov.shape == (4, 4)

    def test_simulate_normal(self, mean_returns, cov_matrix):
        from hangar.risk.engine import RiskEngine
        engine = RiskEngine()
        sims = engine.simulate(mean_returns, cov_matrix, random_state=42)
        assert sims.shape[0] == 10_000
        assert sims.shape[2] == 4

    def test_simulate_student_t(self, mean_returns, cov_matrix):
        from hangar.risk.engine import RiskConfig, RiskEngine
        config = RiskConfig(simulation_method="student_t", simulation_kwargs={"df": 5}, n_sims=100)
        engine = RiskEngine(config)
        sims = engine.simulate(mean_returns, cov_matrix, random_state=42)
        assert sims.shape == (100, 21, 4)

    def test_compute_var_cvar(self, equal_weights, mean_returns, cov_matrix):
        from hangar.risk.engine import RiskConfig, RiskEngine
        config = RiskConfig(n_sims=1000, horizon=5)
        engine = RiskEngine(config)
        result = engine.compute_var_cvar(equal_weights, mean_returns, cov_matrix, random_state=42)
        assert result.cvar >= result.var

    def test_decompose_risk(self, equal_weights, cov_matrix):
        from hangar.risk.engine import RiskEngine
        engine = RiskEngine()
        result = engine.decompose_risk(equal_weights, cov_matrix)
        assert "component_risk" in result
        assert "percent_contribution" in result
        assert "concentration_metrics" in result

    def test_run_stress_test(self, equal_weights, mean_returns, cov_matrix):
        from hangar.risk.engine import RiskConfig, RiskEngine
        config = RiskConfig(n_sims=500, horizon=5)
        engine = RiskEngine(config)
        scenarios = [
            {"name": "vol2x", "type": "vol_spike", "params": {"multiplier": 2.0}},
        ]
        results = engine.run_stress_test(
            equal_weights, mean_returns, cov_matrix, scenarios, random_state=42
        )
        assert len(results) == 1
        assert results[0]["name"] == "vol2x"


class TestRiskConfigLoaders:
    def test_load_risk_config(self):
        from hangar.risk.engine import load_risk_config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"covariance_method": "oas", "n_sims": 5000}, f)
            f.flush()
            config = load_risk_config(f.name)
        os.unlink(f.name)
        assert config.covariance_method == "oas"
        assert config.n_sims == 5000

    def test_load_scenarios_config(self):
        from hangar.risk.engine import load_scenarios_config
        data = {"scenarios": [{"name": "test", "type": "vol_spike", "params": {}}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = load_scenarios_config(f.name)
        os.unlink(f.name)
        assert len(result) == 1
        assert result[0]["name"] == "test"

    def test_load_actual_scenarios_config(self):
        from hangar.risk.engine import load_scenarios_config
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "risk", "scenarios.json"
        )
        if os.path.exists(path):
            scenarios = load_scenarios_config(path)
            assert len(scenarios) >= 1


# ===================================================================
# hangar.risk.__init__ re-exports
# ===================================================================

class TestRiskH2PublicApi:
    def test_all_h2_exports_importable(self):
        import hangar.risk as risk_mod
        h2_names = [
            "RiskConfig", "RiskEngine", "ScenarioResult", "SimulationMethod",
            "apply_vol_spike", "apply_correlation_spike", "apply_sector_crash",
            "cluster_risk_attribution", "component_risk", "concentration_metrics",
            "expanding_regime_covariance", "factor_risk_decomposition",
            "load_risk_config", "load_scenarios_config",
            "marginal_contribution_to_risk", "percent_risk_contribution",
            "regime_aware_covariance", "run_scenario", "run_scenario_suite",
            "scenario_report_to_dict", "simulate_factor_returns",
            "simulate_returns", "simulate_student_t_returns",
        ]
        for name in h2_names:
            assert hasattr(risk_mod, name), f"{name} not importable from hangar.risk"
