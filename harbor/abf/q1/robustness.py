"""harbor.abf.q1.robustness — Robustness sweep for ABF Q1 local projections.

Provides alternative shock definitions, sample splits, and a full
combinatorial sweep producing a summary DataFrame of coefficient estimates,
standard errors, p-values, and fit statistics.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from harbor.abf.q1.analysis import fit_local_projection
from harbor.risk.regime_detection import detect_vol_shocks


def apply_shock_definition(
    returns: pd.Series,
    config_dict: Dict[str, Any],
) -> pd.Series:
    """Dispatch to a shock detection method based on a config dict.

    Supported methods:
    - ``realized_vol_change_percentile``: delegates to ``detect_vol_shocks``
    - ``realized_vol_level_percentile``: flags dates where rolling vol exceeds
      a percentile threshold
    - ``range_based_volatility_jump``: uses rolling range-based vol proxy
    - ``vix_jump``: warns and falls back to ``realized_vol_change_percentile``

    Parameters
    ----------
    returns
        Daily return series.
    config_dict
        Must contain ``"method"`` plus method-specific keys.

    Returns
    -------
    pd.Series[bool]
        Boolean shock indicator aligned to ``returns.index``.
    """
    method = config_dict.get("method", "")
    threshold = config_dict.get("threshold_percentile", 0.95)
    lookback = config_dict.get("lookback_days", 21)

    if method == "realized_vol_change_percentile":
        return detect_vol_shocks(returns, threshold_pct=threshold, vol_window=lookback)

    if method == "realized_vol_level_percentile":
        rv = returns.rolling(lookback).std()
        cutoff = rv.quantile(threshold)
        shocks = (rv >= cutoff) & rv.notna()
        shocks.name = "vol_shock"
        return shocks.astype(bool)

    if method in ("range_based_volatility_jump", "range_based"):
        # Use absolute return as a range-based vol proxy
        abs_ret = returns.abs()
        rv_proxy = abs_ret.rolling(lookback).mean()
        rv_change = rv_proxy.diff().abs()
        cutoff = rv_change.quantile(threshold)
        shocks = (rv_change >= cutoff) & rv_change.notna()
        shocks.name = "vol_shock"
        return shocks.astype(bool)

    if method == "vix_jump":
        warnings.warn(
            "vix_jump requires external VIX data; falling back to "
            "realized_vol_change_percentile.",
            stacklevel=2,
        )
        return detect_vol_shocks(returns, threshold_pct=threshold, vol_window=lookback)

    raise ValueError(f"Unknown shock method: {method!r}")


def split_sample(
    returns: pd.Series,
    split_name: str,
) -> pd.Series:
    """Return a boolean mask selecting a sample sub-period.

    Supported splits:
    - ``full``: all dates
    - ``pre_2020``: before 2020-01-01
    - ``post_2020``: 2020-01-01 onwards
    - ``high_liquidity``: above-median 21-day average absolute return
    - ``low_liquidity``: below-median 21-day average absolute return

    Parameters
    ----------
    returns
        Daily return series.
    split_name
        Name of the sample split.

    Returns
    -------
    pd.Series[bool]
        Boolean mask aligned to ``returns.index``.
    """
    idx = returns.index

    if split_name == "full":
        return pd.Series(True, index=idx, name="sample_mask")

    if split_name == "pre_2020":
        mask = idx < pd.Timestamp("2020-01-01")
        return pd.Series(mask, index=idx, name="sample_mask")

    if split_name == "post_2020":
        mask = idx >= pd.Timestamp("2020-01-01")
        return pd.Series(mask, index=idx, name="sample_mask")

    if split_name == "high_liquidity":
        liq = returns.abs().rolling(21).mean()
        median_liq = liq.median()
        mask = liq >= median_liq
        return mask.fillna(False).rename("sample_mask")

    if split_name == "low_liquidity":
        liq = returns.abs().rolling(21).mean()
        median_liq = liq.median()
        mask = liq < median_liq
        return mask.fillna(False).rename("sample_mask")

    raise ValueError(f"Unknown sample split: {split_name!r}")


def robustness_sweep(
    returns: pd.Series,
    shock_configs: List[Dict[str, Any]],
    sample_splits: List[str],
    horizons: List[int],
    vol_proxy: pd.Series,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    """Run local projections across all shock × split × horizon combos.

    Parameters
    ----------
    returns
        Daily return series.
    shock_configs
        List of shock definition config dicts (each has ``"method"`` key).
    sample_splits
        List of split names (e.g. ``["full", "pre_2020", "post_2020"]``).
    horizons
        Forward horizons in trading days.
    vol_proxy
        Vol-control pressure proxy.
    controls
        Control matrix from ``build_control_matrix``.

    Returns
    -------
    pd.DataFrame
        One row per (shock_method, sample_split, horizon) with columns:
        ``shock_method``, ``sample_split``, ``horizon``, ``b_h``, ``b_h_se``,
        ``b_h_pval``, ``c_h``, ``c_h_se``, ``c_h_pval``, ``r_squared``,
        ``n_obs``.
    """
    rows: List[Dict[str, Any]] = []

    for cfg in shock_configs:
        method_name = cfg.get("method", "unknown")
        shocks = apply_shock_definition(returns, cfg)

        for split_name in sample_splits:
            mask = split_sample(returns, split_name)

            # Apply mask
            ret_sub = returns[mask]
            shocks_sub = shocks[mask]
            proxy_sub = vol_proxy[mask]
            ctrl_sub = controls.loc[mask.index[mask]]

            if ret_sub.shape[0] < 50:
                for h in horizons:
                    rows.append(_empty_row(method_name, split_name, h, ret_sub.shape[0]))
                continue

            for h in horizons:
                try:
                    result = fit_local_projection(ret_sub, shocks_sub, proxy_sub, ctrl_sub, h)
                    b_h = _safe_param(result, "shock")
                    b_h_se = _safe_se(result, "shock")
                    b_h_pval = _safe_pval(result, "shock")
                    c_h = _safe_param(result, "shock_x_vol_proxy")
                    c_h_se = _safe_se(result, "shock_x_vol_proxy")
                    c_h_pval = _safe_pval(result, "shock_x_vol_proxy")

                    rows.append(
                        {
                            "shock_method": method_name,
                            "sample_split": split_name,
                            "horizon": h,
                            "b_h": b_h,
                            "b_h_se": b_h_se,
                            "b_h_pval": b_h_pval,
                            "c_h": c_h,
                            "c_h_se": c_h_se,
                            "c_h_pval": c_h_pval,
                            "r_squared": result.rsquared,
                            "n_obs": int(result.nobs),
                        }
                    )
                except (ValueError, np.linalg.LinAlgError):
                    rows.append(_empty_row(method_name, split_name, h, 0))

    return pd.DataFrame(rows)


def _safe_param(result, name: str) -> float:
    if name in result.params.index:
        return float(result.params[name])
    return float("nan")


def _safe_se(result, name: str) -> float:
    if name in result.bse.index:
        return float(result.bse[name])
    return float("nan")


def _safe_pval(result, name: str) -> float:
    if name in result.pvalues.index:
        return float(result.pvalues[name])
    return float("nan")


def _empty_row(method: str, split: str, horizon: int, n_obs: int) -> Dict[str, Any]:
    return {
        "shock_method": method,
        "sample_split": split,
        "horizon": horizon,
        "b_h": float("nan"),
        "b_h_se": float("nan"),
        "b_h_pval": float("nan"),
        "c_h": float("nan"),
        "c_h_se": float("nan"),
        "c_h_pval": float("nan"),
        "r_squared": float("nan"),
        "n_obs": n_obs,
    }
