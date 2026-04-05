"""Risk models, covariance estimators, and simulation utilities."""

from harbor.risk.correlation import detect_correlation_spikes
from harbor.risk.covariance import (
    CovarianceMethod,
    estimate_covariance,
    expanding_regime_covariance,
    regime_aware_covariance,
    sample_covariance,
    shrinkage_covariance,
)
from harbor.risk.decomposition import (
    cluster_risk_attribution,
    component_risk,
    concentration_metrics,
    factor_risk_decomposition,
    marginal_contribution_to_risk,
    percent_risk_contribution,
)
from harbor.risk.engine import RiskConfig, RiskEngine, load_risk_config, load_scenarios_config
from harbor.risk.hrp import hrp_allocation
from harbor.risk.monte_carlo import (
    SimulationMethod,
    VarCvarResult,
    monte_carlo_var_cvar_from_history,
    portfolio_var_cvar,
    simulate_factor_returns,
    simulate_multivariate_returns,
    simulate_returns,
    simulate_student_t_returns,
)
from harbor.risk.regime_detection import detect_vol_shocks, vol_control_pressure_proxy
from harbor.risk.scenarios import (
    ScenarioResult,
    apply_correlation_spike,
    apply_sector_crash,
    apply_vol_spike,
    run_scenario,
    run_scenario_suite,
    scenario_report_to_dict,
)

__all__ = [
    "CovarianceMethod",
    "RiskConfig",
    "RiskEngine",
    "ScenarioResult",
    "SimulationMethod",
    "VarCvarResult",
    "apply_correlation_spike",
    "apply_sector_crash",
    "apply_vol_spike",
    "cluster_risk_attribution",
    "component_risk",
    "concentration_metrics",
    "detect_correlation_spikes",
    "detect_vol_shocks",
    "estimate_covariance",
    "expanding_regime_covariance",
    "factor_risk_decomposition",
    "hrp_allocation",
    "load_risk_config",
    "load_scenarios_config",
    "marginal_contribution_to_risk",
    "monte_carlo_var_cvar_from_history",
    "percent_risk_contribution",
    "portfolio_var_cvar",
    "regime_aware_covariance",
    "run_scenario",
    "run_scenario_suite",
    "sample_covariance",
    "scenario_report_to_dict",
    "shrinkage_covariance",
    "simulate_factor_returns",
    "simulate_multivariate_returns",
    "simulate_returns",
    "simulate_student_t_returns",
    "vol_control_pressure_proxy",
]
