"""hangar.ml.volatility.integration — Bridge vol forecasts into hangar allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_scaled_weight_func(
    base_weight_func: WeightFunction,  # noqa: F821
    sigma_hat: pd.Series,
    *,
    target_vol: float = 0.10,
    floor: float = 0.2,
    cap: float = 2.0,
) -> WeightFunction:  # noqa: F821
    """Wrap a base WeightFunction with volatility-targeting scaling.

    The returned function computes base weights, then scales them by
    ``target_vol / sigma_hat_t``, clipped to ``[floor, cap]``.

    Parameters
    ----------
    base_weight_func
        Any existing WeightFunction (e.g., risk parity, mean-variance).
    sigma_hat
        Predicted volatility series from a trained forecaster.
        Must have a DatetimeIndex covering the backtest period.
    target_vol
        Annualized target portfolio volatility.
    floor
        Minimum scaling factor (prevents extreme leverage).
    cap
        Maximum scaling factor.

    Returns
    -------
    WeightFunction
        A new weight function compatible with ``run_cross_sectional_backtest``.
    """

    def _scaled(lookback: pd.DataFrame, current_weights: pd.Series) -> pd.Series:
        base_weights = base_weight_func(lookback, current_weights)

        # Find the current date from the lookback window
        current_date = lookback.index[-1]

        # Look up sigma_hat — use nearest available date if exact match missing
        if current_date in sigma_hat.index:
            sigma = sigma_hat.loc[current_date]
        else:
            idx = sigma_hat.index.get_indexer([current_date], method="ffill")
            if idx[0] == -1:
                return base_weights
            sigma = sigma_hat.iloc[idx[0]]

        if sigma <= 0 or np.isnan(sigma):
            return base_weights

        scale = np.clip(target_vol / sigma, floor, cap)
        scaled = base_weights * scale

        # Re-normalize to sum to 1 (long-only)
        total = scaled.sum()
        if total > 0:
            scaled = scaled / total
        return scaled

    return _scaled


def sigma_hat_to_regime_proxy(
    sigma_hat: pd.Series,
    *,
    percentile_threshold: float = 0.80,
    rolling_window: int = 252,
) -> pd.Series:
    """Convert predicted volatility into a [0, 1] shock proxy.

    Values above the ``percentile_threshold`` of the trailing distribution
    are mapped to elevated shock levels.  Compatible with
    ``hangar.portfolio.regime_aware_position_size``.

    Parameters
    ----------
    sigma_hat
        Predicted volatility series.
    percentile_threshold
        Percentile above which vol is considered elevated.
    rolling_window
        Window for computing the rolling percentile rank.

    Returns
    -------
    pd.Series
        Shock proxy in [0, 1], indexed by the same dates as ``sigma_hat``.
    """

    def _rolling_pctrank(s: pd.Series, window: int) -> pd.Series:
        result = pd.Series(np.nan, index=s.index, name="regime_proxy")
        values = s.values
        for i in range(window, len(values)):
            window_vals = values[i - window : i + 1]
            current = values[i]
            rank = np.sum(window_vals <= current) / len(window_vals)
            result.iloc[i] = rank
        return result

    pct_rank = _rolling_pctrank(sigma_hat, rolling_window)

    # Map: below threshold → 0, above threshold → linearly scaled to [0, 1]
    proxy = (pct_rank - percentile_threshold) / (1.0 - percentile_threshold)
    proxy = proxy.clip(lower=0.0, upper=1.0)
    proxy.name = "regime_proxy"
    return proxy
