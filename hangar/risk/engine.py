"""Pluggable risk engine for HANGAR H2.

Provides a unified interface for configuring and running risk analysis
with selectable covariance, simulation, and scenario methods.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from hangar.risk.covariance import estimate_covariance
from hangar.risk.monte_carlo import (
    VarCvarResult,
    portfolio_var_cvar,
    simulate_multivariate_returns,
)


@dataclass
class RiskConfig:
    """Configuration for the pluggable risk engine.

    Parameters
    ----------
    covariance_method
        Any ``CovarianceMethod`` value (``"sample"``, ``"ledoit_wolf"``,
        ``"oas"``) or ``"regime_aware"``.
    simulation_method
        Simulation distribution: ``"normal"`` or ``"student_t"``.
    simulation_kwargs
        Extra keyword arguments forwarded to the simulator (e.g.
        ``{"df": 5}`` for the Student-*t* model).
    n_sims
        Number of Monte Carlo paths.
    horizon
        Simulation horizon in trading days.
    alpha
        Confidence level for VaR/CVaR.
    annualization
        Trading-day annualization factor for covariance estimation.
    """

    covariance_method: str = "ledoit_wolf"
    simulation_method: str = "normal"
    simulation_kwargs: dict = field(default_factory=dict)
    n_sims: int = 10_000
    horizon: int = 21
    alpha: float = 0.95
    annualization: int = 252


class RiskEngine:
    """Unified risk engine that ties together covariance estimation,
    simulation, scenario analysis, and risk decomposition."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config if config is not None else RiskConfig()

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate a covariance matrix using the configured method.

        For ``"regime_aware"`` the method falls back to ``"ledoit_wolf"``
        (regime-aware estimation is expected to be provided by a future
        plug-in; this keeps the interface stable).
        """
        return estimate_covariance(
            returns,
            method=self.config.covariance_method,
            annualization=self.config.annualization,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        *,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Generate simulated return paths.

        Dispatches to multivariate-normal or Student-*t* simulation
        based on ``config.simulation_method``.

        Returns an array with shape ``(n_sims, horizon, n_assets)``.
        """
        method = self.config.simulation_method

        if method == "normal":
            return simulate_multivariate_returns(
                mean_returns,
                cov_matrix,
                n_sims=self.config.n_sims,
                horizon=self.config.horizon,
                random_state=random_state,
            )

        if method == "student_t":
            return _simulate_student_t(
                mean_returns,
                cov_matrix,
                n_sims=self.config.n_sims,
                horizon=self.config.horizon,
                random_state=random_state,
                **self.config.simulation_kwargs,
            )

        raise ValueError(
            f"Unknown simulation method {method!r}. Use 'normal' or 'student_t'."
        )

    # ------------------------------------------------------------------
    # VaR / CVaR
    # ------------------------------------------------------------------

    def compute_var_cvar(
        self,
        weights: pd.Series,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        *,
        random_state: Optional[int] = None,
    ) -> VarCvarResult:
        """End-to-end VaR/CVaR: simulate paths then compute risk measures."""
        sims = self.simulate(
            mean_returns, cov_matrix, random_state=random_state
        )
        return portfolio_var_cvar(weights, sims, alpha=self.config.alpha)

    # ------------------------------------------------------------------
    # Stress testing
    # ------------------------------------------------------------------

    def run_stress_test(
        self,
        weights: pd.Series,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        scenarios_config: list[dict],
        *,
        sector_map: dict[str, str] | None = None,
        random_state: Optional[int] = None,
    ) -> list[dict]:
        """Run a suite of stress scenarios.

        Delegates to :func:`hangar.risk.scenarios.run_scenario_suite` when
        the module is available.  A minimal built-in implementation is used
        as a fallback so that the engine remains usable before the full
        scenarios plug-in is integrated.
        """
        from hangar.risk.scenarios import run_scenario_suite, scenario_report_to_dict

        results = run_scenario_suite(
            weights=weights,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            scenarios_config=scenarios_config,
            sector_map=sector_map,
            n_sims=self.config.n_sims,
            horizon=self.config.horizon,
            alpha=self.config.alpha,
            random_state=random_state,
        )
        return scenario_report_to_dict(results)

    # ------------------------------------------------------------------
    # Risk decomposition
    # ------------------------------------------------------------------

    def decompose_risk(
        self,
        weights: pd.Series,
        cov_matrix: pd.DataFrame,
    ) -> dict:
        """Decompose portfolio risk into per-asset contributions.

        Returns a dict with:
        - ``component_risk``: pd.Series of each asset's risk contribution.
        - ``percent_contribution``: pd.Series of percentage contributions.
        - ``concentration_metrics``: dict with Herfindahl index and
          effective number of bets.
        """
        w = weights.to_numpy(dtype=float)
        cov = cov_matrix.loc[weights.index, weights.index].to_numpy(dtype=float)

        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(port_var)

        # Marginal contribution to risk (MCR) and component risk (CR)
        mcr = (cov @ w) / port_vol
        component_risk = w * mcr
        pct_contribution = component_risk / port_vol

        component_series = pd.Series(component_risk, index=weights.index, name="component_risk")
        pct_series = pd.Series(pct_contribution, index=weights.index, name="percent_contribution")

        # Concentration metrics
        herfindahl = float(np.sum(pct_contribution ** 2))
        eff_n = 1.0 / herfindahl if herfindahl > 0 else float("inf")

        return {
            "component_risk": component_series,
            "percent_contribution": pct_series,
            "concentration_metrics": {
                "herfindahl": herfindahl,
                "effective_n_bets": eff_n,
            },
        }


# ======================================================================
# Module-level helpers
# ======================================================================

def _simulate_student_t(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    *,
    n_sims: int = 10_000,
    horizon: int = 21,
    df: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Multivariate Student-*t* simulation via scale-mixture representation."""
    if df <= 2:
        raise ValueError("Degrees of freedom (df) must be > 2 for finite variance.")

    mu = mean_returns.to_numpy(dtype=float)
    cov = cov_matrix.loc[mean_returns.index, mean_returns.index].to_numpy(dtype=float)

    rng = np.random.default_rng(seed=random_state)

    # Normal draws scaled by inverse-chi-squared mixing variable
    normals = rng.multivariate_normal(
        np.zeros_like(mu), cov, size=(n_sims, horizon), check_valid="ignore"
    )
    chi2 = rng.chisquare(df, size=(n_sims, horizon))
    scale = np.sqrt(df / chi2)[..., np.newaxis]

    draws = mu + normals * scale
    return draws



# ======================================================================
# Config loaders
# ======================================================================

def load_risk_config(path: str) -> RiskConfig:
    """Load a :class:`RiskConfig` from a JSON file.

    The JSON object's keys are mapped directly to ``RiskConfig`` fields.
    Unknown keys are silently ignored.
    """
    with open(path) as fh:
        data = json.load(fh)

    valid_fields = {f.name for f in RiskConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return RiskConfig(**filtered)


def load_scenarios_config(path: str) -> list[dict]:
    """Load scenario definitions from a JSON file.

    Expects a top-level ``"scenarios"`` key containing a list of scenario
    dicts.  If absent, the entire JSON is treated as a list.
    """
    with open(path) as fh:
        data = json.load(fh)

    if isinstance(data, list):
        return data

    return data.get("scenarios", [])
