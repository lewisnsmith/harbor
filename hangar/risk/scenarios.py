"""Stress scenario runner for HANGAR H2.

Provides config-driven stress testing with vol spike, correlation spike,
and sector crash scenarios. Produces machine-readable report dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from hangar.risk.monte_carlo import portfolio_var_cvar, simulate_multivariate_returns


@dataclass
class ScenarioResult:
    """Container for a single stress scenario outcome."""

    name: str
    description: str
    stressed_cov: pd.DataFrame
    baseline_var: float
    stressed_var: float
    baseline_cvar: float
    stressed_cvar: float
    var_change_pct: float
    cvar_change_pct: float


# ---------------------------------------------------------------------------
# Scenario transforms
# ---------------------------------------------------------------------------


def apply_vol_spike(cov_matrix: pd.DataFrame, multiplier: float = 2.0) -> pd.DataFrame:
    """Scale all variances (diagonal) by *multiplier*, keeping correlations fixed.

    Parameters
    ----------
    cov_matrix:
        Covariance matrix as a square DataFrame.
    multiplier:
        Factor by which to scale variances.  Must be > 0.

    Returns
    -------
    pd.DataFrame
        New covariance matrix with spiked volatilities.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")

    cov = cov_matrix.to_numpy(dtype=float)
    stds = np.sqrt(np.diag(cov))

    # Correlation matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)

    # Scale standard deviations
    new_stds = stds * np.sqrt(multiplier)
    new_cov = corr * np.outer(new_stds, new_stds)

    return pd.DataFrame(new_cov, index=cov_matrix.index, columns=cov_matrix.columns)


def apply_correlation_spike(
    cov_matrix: pd.DataFrame, target_corr: float = 0.95
) -> pd.DataFrame:
    """Set all off-diagonal correlations to *target_corr*, keeping variances unchanged.

    Parameters
    ----------
    cov_matrix:
        Covariance matrix as a square DataFrame.
    target_corr:
        Target pairwise correlation for every off-diagonal entry.  Must be
        in (-1, 1].

    Returns
    -------
    pd.DataFrame
        New covariance matrix with uniform off-diagonal correlations.
    """
    if not (-1.0 < target_corr <= 1.0):
        raise ValueError("target_corr must be in (-1, 1]")

    cov = cov_matrix.to_numpy(dtype=float)
    stds = np.sqrt(np.diag(cov))
    n = len(stds)

    corr = np.full((n, n), target_corr, dtype=float)
    np.fill_diagonal(corr, 1.0)

    new_cov = corr * np.outer(stds, stds)
    return pd.DataFrame(new_cov, index=cov_matrix.index, columns=cov_matrix.columns)


def apply_sector_crash(
    mean_returns: pd.Series,
    sector_map: dict[str, str],
    crash_sector: str,
    crash_magnitude: float = -0.15,
) -> pd.Series:
    """Apply a return shock to every asset in *crash_sector*.

    Parameters
    ----------
    mean_returns:
        Expected daily mean returns per asset.
    sector_map:
        Mapping of asset name -> sector label.
    crash_sector:
        Sector to shock.
    crash_magnitude:
        Additive return shock applied to each asset in the sector
        (e.g. -0.15 means -15%).

    Returns
    -------
    pd.Series
        Modified mean returns with the crash applied.

    Raises
    ------
    ValueError
        If *crash_sector* is not found in *sector_map* values.
    """
    available_sectors = set(sector_map.values())
    if crash_sector not in available_sectors:
        raise ValueError(
            f"crash_sector {crash_sector!r} not found in sector_map. "
            f"Available sectors: {sorted(available_sectors)}"
        )

    shocked = mean_returns.copy()
    for asset, sector in sector_map.items():
        if sector == crash_sector and asset in shocked.index:
            shocked[asset] += crash_magnitude
    return shocked


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------


def run_scenario(
    name: str,
    description: str,
    weights: pd.Series,
    mean_returns: pd.Series,
    baseline_cov: pd.DataFrame,
    stressed_cov: Optional[pd.DataFrame] = None,
    stressed_mean: Optional[pd.Series] = None,
    *,
    n_sims: int = 10_000,
    horizon: int = 21,
    alpha: float = 0.95,
    random_state: Optional[int] = None,
) -> ScenarioResult:
    """Run a single stress scenario and return baseline vs stressed metrics.

    Parameters
    ----------
    name:
        Short identifier for the scenario.
    description:
        Human-readable description.
    weights:
        Portfolio weights.
    mean_returns:
        Baseline expected mean returns.
    baseline_cov:
        Baseline covariance matrix.
    stressed_cov:
        Stressed covariance matrix.  If *None*, ``baseline_cov`` is used.
    stressed_mean:
        Stressed mean returns.  If *None*, ``mean_returns`` is used.
    n_sims:
        Number of Monte Carlo simulations.
    horizon:
        Simulation horizon in trading days.
    alpha:
        VaR/CVaR confidence level.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    ScenarioResult
    """
    if stressed_cov is None:
        stressed_cov = baseline_cov
    if stressed_mean is None:
        stressed_mean = mean_returns

    # Baseline simulation
    baseline_sims = simulate_multivariate_returns(
        mean_returns,
        baseline_cov,
        n_sims=n_sims,
        horizon=horizon,
        random_state=random_state,
    )
    baseline_risk = portfolio_var_cvar(weights, baseline_sims, alpha=alpha)

    # Stressed simulation (use same seed for comparability)
    stressed_sims = simulate_multivariate_returns(
        stressed_mean,
        stressed_cov,
        n_sims=n_sims,
        horizon=horizon,
        random_state=random_state,
    )
    stressed_risk = portfolio_var_cvar(weights, stressed_sims, alpha=alpha)

    # Percentage changes (guard against zero baseline)
    def _pct_change(baseline: float, stressed: float) -> float:
        if baseline == 0.0:
            return 0.0 if stressed == 0.0 else float("inf")
        return (stressed - baseline) / abs(baseline) * 100.0

    return ScenarioResult(
        name=name,
        description=description,
        stressed_cov=stressed_cov,
        baseline_var=baseline_risk.var,
        stressed_var=stressed_risk.var,
        baseline_cvar=baseline_risk.cvar,
        stressed_cvar=stressed_risk.cvar,
        var_change_pct=_pct_change(baseline_risk.var, stressed_risk.var),
        cvar_change_pct=_pct_change(baseline_risk.cvar, stressed_risk.cvar),
    )


def run_scenario_suite(
    weights: pd.Series,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    scenarios_config: list[dict[str, Any]],
    *,
    sector_map: Optional[dict[str, str]] = None,
    n_sims: int = 10_000,
    horizon: int = 21,
    alpha: float = 0.95,
    random_state: Optional[int] = None,
) -> list[ScenarioResult]:
    """Run multiple stress scenarios from a configuration list.

    Parameters
    ----------
    weights:
        Portfolio weights.
    mean_returns:
        Baseline expected mean returns.
    cov_matrix:
        Baseline covariance matrix.
    scenarios_config:
        List of scenario specification dicts.  Each dict must contain:

        - ``name`` (str): scenario identifier
        - ``type`` (str): one of ``"vol_spike"``, ``"correlation_spike"``,
          ``"sector_crash"``
        - ``params`` (dict): keyword arguments forwarded to the
          corresponding ``apply_*`` function.

        Example::

            [
                {"name": "vol_spike_2x", "type": "vol_spike",
                 "params": {"multiplier": 2.0}},
                {"name": "corr_spike_95", "type": "correlation_spike",
                 "params": {"target_corr": 0.95}},
                {"name": "tech_crash", "type": "sector_crash",
                 "params": {"crash_sector": "Technology",
                            "crash_magnitude": -0.15}},
            ]

    sector_map:
        Mapping of asset name -> sector label.  Required when any scenario
        has ``type="sector_crash"``.
    n_sims:
        Number of Monte Carlo simulations per scenario.
    horizon:
        Simulation horizon in trading days.
    alpha:
        VaR/CVaR confidence level.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    list[ScenarioResult]
    """
    _SCENARIO_TYPES = {"vol_spike", "correlation_spike", "sector_crash"}
    results: list[ScenarioResult] = []

    for cfg in scenarios_config:
        name = cfg.get("name")
        scenario_type = cfg.get("type")
        params: dict[str, Any] = cfg.get("params", {})

        if name is None:
            raise ValueError("Each scenario config must include a 'name' key.")
        if scenario_type not in _SCENARIO_TYPES:
            raise ValueError(
                f"Unknown scenario type {scenario_type!r} for scenario {name!r}. "
                f"Must be one of {sorted(_SCENARIO_TYPES)}."
            )

        stressed_cov: Optional[pd.DataFrame] = None
        stressed_mean: Optional[pd.Series] = None
        description: str

        if scenario_type == "vol_spike":
            multiplier = params.get("multiplier", 2.0)
            stressed_cov = apply_vol_spike(cov_matrix, multiplier=multiplier)
            description = f"Volatility spike: variances scaled by {multiplier}x"

        elif scenario_type == "correlation_spike":
            target_corr = params.get("target_corr", 0.95)
            stressed_cov = apply_correlation_spike(cov_matrix, target_corr=target_corr)
            description = f"Correlation spike: off-diagonal correlations set to {target_corr}"

        elif scenario_type == "sector_crash":
            if sector_map is None:
                raise ValueError(
                    f"sector_map is required for sector_crash scenario {name!r}."
                )
            crash_sector = params.get("crash_sector")
            if crash_sector is None:
                raise ValueError(
                    f"'crash_sector' param is required for scenario {name!r}."
                )
            crash_magnitude = params.get("crash_magnitude", -0.15)
            stressed_mean = apply_sector_crash(
                mean_returns,
                sector_map,
                crash_sector=crash_sector,
                crash_magnitude=crash_magnitude,
            )
            description = (
                f"Sector crash: {crash_sector} shocked by "
                f"{crash_magnitude:+.0%}"
            )

        result = run_scenario(
            name=name,
            description=description,
            weights=weights,
            mean_returns=mean_returns,
            baseline_cov=cov_matrix,
            stressed_cov=stressed_cov,
            stressed_mean=stressed_mean,
            n_sims=n_sims,
            horizon=horizon,
            alpha=alpha,
            random_state=random_state,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def scenario_report_to_dict(results: list[ScenarioResult]) -> list[dict[str, Any]]:
    """Convert a list of :class:`ScenarioResult` to plain dicts for JSON serialization.

    DataFrames are converted to nested dicts via :meth:`DataFrame.to_dict`.

    Parameters
    ----------
    results:
        Scenario results to serialize.

    Returns
    -------
    list[dict[str, Any]]
    """
    out: list[dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "name": r.name,
                "description": r.description,
                "stressed_cov": r.stressed_cov.to_dict(),
                "baseline_var": r.baseline_var,
                "stressed_var": r.stressed_var,
                "baseline_cvar": r.baseline_cvar,
                "stressed_cvar": r.stressed_cvar,
                "var_change_pct": r.var_change_pct,
                "cvar_change_pct": r.cvar_change_pct,
            }
        )
    return out
