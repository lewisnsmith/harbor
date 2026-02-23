"""Risk models, covariance estimators, and simulation utilities."""

from harbor.risk.correlation import detect_correlation_spikes
from harbor.risk.covariance import (
    estimate_covariance,
    sample_covariance,
    shrinkage_covariance,
)
from harbor.risk.hrp import hrp_allocation
from harbor.risk.monte_carlo import (
    VarCvarResult,
    monte_carlo_var_cvar_from_history,
    portfolio_var_cvar,
    simulate_multivariate_returns,
)
from harbor.risk.regime_detection import detect_vol_shocks, vol_control_pressure_proxy

__all__ = [
    "VarCvarResult",
    "detect_correlation_spikes",
    "detect_vol_shocks",
    "estimate_covariance",
    "hrp_allocation",
    "monte_carlo_var_cvar_from_history",
    "portfolio_var_cvar",
    "sample_covariance",
    "shrinkage_covariance",
    "simulate_multivariate_returns",
    "vol_control_pressure_proxy",
]
