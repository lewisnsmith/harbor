"""harbor.abf.q1.visualization — Figure generation for ABF Q1 analysis.

All functions return ``matplotlib.figure.Figure`` instances and do not
call ``plt.show()``, so callers can save or display as needed.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_shock_timeline(
    returns: pd.Series,
    shocks: pd.Series,
    vol_proxy: pd.Series,
) -> plt.Figure:
    """Two-panel timeline: cumulative returns and vol proxy with shock markers.

    Parameters
    ----------
    returns
        Daily return series.
    shocks
        Boolean shock indicator.
    vol_proxy
        Vol-control pressure proxy.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    cum_ret = (1 + returns).cumprod() - 1
    ax1.plot(cum_ret.index, cum_ret.values, color="steelblue", linewidth=0.8)

    shock_dates = shocks.index[shocks]
    for d in shock_dates:
        ax1.axvline(d, color="red", alpha=0.15, linewidth=0.5)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Cumulative Returns with Shock Dates")
    ax1.grid(True, alpha=0.3)

    proxy = vol_proxy.reindex(returns.index)
    ax2.plot(proxy.index, proxy.values, color="darkorange", linewidth=0.8)
    for d in shock_dates:
        ax2.axvline(d, color="red", alpha=0.15, linewidth=0.5)
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Vol-Control Pressure Proxy")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_car_paths(
    car_results: Dict[int, pd.DataFrame],
    horizons: List[int] = None,
) -> plt.Figure:
    """CAR paths with SE bands per horizon.

    Parameters
    ----------
    car_results
        Mapping ``horizon -> DataFrame`` from ``cumulative_abnormal_return``.
        Each DataFrame has a ``"car"`` column.
    horizons
        Horizons to plot. Defaults to all keys in ``car_results``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if horizons is None:
        horizons = sorted(car_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(horizons), 1)))

    for i, h in enumerate(horizons):
        df = car_results[h]
        if df.empty:
            continue
        car_vals = df["car"]
        mean_car = float(car_vals.mean())
        se_car = float(car_vals.std() / np.sqrt(len(car_vals))) if len(car_vals) > 1 else 0.0

        ax.bar(
            i,
            mean_car,
            yerr=1.96 * se_car,
            color=colors[i],
            capsize=5,
            label=f"h={h}",
        )

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("Mean CAR")
    ax.set_title("Cumulative Abnormal Returns by Horizon")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_coefficient_path(
    lp_results: Dict[int, object],
    horizons: List[int] = None,
) -> plt.Figure:
    """Plot shock (b_h) and interaction (c_h) coefficients across horizons with 95% CI.

    Parameters
    ----------
    lp_results
        Mapping ``horizon -> statsmodels RegressionResults``.
    horizons
        Horizons to include. Defaults to sorted keys.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if horizons is None:
        horizons = sorted(lp_results.keys())

    b_vals, b_ci_lo, b_ci_hi = [], [], []
    c_vals, c_ci_lo, c_ci_hi = [], [], []

    for h in horizons:
        res = lp_results[h]
        params = res.params
        bse = res.bse

        b = params.get("shock", float("nan"))
        b_se = bse.get("shock", 0.0)
        b_vals.append(b)
        b_ci_lo.append(b - 1.96 * b_se)
        b_ci_hi.append(b + 1.96 * b_se)

        c = params.get("shock_x_vol_proxy", float("nan"))
        c_se = bse.get("shock_x_vol_proxy", 0.0)
        c_vals.append(c)
        c_ci_lo.append(c - 1.96 * c_se)
        c_ci_hi.append(c + 1.96 * c_se)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(horizons))

    ax1.errorbar(
        x,
        b_vals,
        yerr=[np.array(b_vals) - np.array(b_ci_lo), np.array(b_ci_hi) - np.array(b_vals)],
        fmt="o-",
        capsize=5,
        color="steelblue",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(h) for h in horizons])
    ax1.set_xlabel("Horizon (days)")
    ax1.set_ylabel("Coefficient")
    ax1.set_title(r"$\beta_h$ (Shock effect)")
    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(
        x,
        c_vals,
        yerr=[np.array(c_vals) - np.array(c_ci_lo), np.array(c_ci_hi) - np.array(c_vals)],
        fmt="s-",
        capsize=5,
        color="darkorange",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(h) for h in horizons])
    ax2.set_xlabel("Horizon (days)")
    ax2.set_ylabel("Coefficient")
    ax2.set_title(r"$\gamma_h$ (Shock $\times$ VolProxy)")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_robustness_heatmap(
    sweep_results: pd.DataFrame,
) -> plt.Figure:
    """Heatmap of interaction coefficient significance across shock × split combos.

    Displays one sub-heatmap per horizon, with shock_method on y-axis and
    sample_split on x-axis. Cell values are ``-log10(c_h_pval)``, capped at 5.

    Parameters
    ----------
    sweep_results
        DataFrame from ``robustness_sweep``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    horizons = sorted(sweep_results["horizon"].unique())
    n_h = len(horizons)

    fig, axes = plt.subplots(1, max(n_h, 1), figsize=(6 * max(n_h, 1), 5))
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        sub = sweep_results[sweep_results["horizon"] == h]
        pivot = sub.pivot(index="shock_method", columns="sample_split", values="c_h_pval")
        # Transform: -log10(pval), capped at 5
        display = -np.log10(pivot.clip(lower=1e-10)).clip(upper=5)

        im = ax.imshow(display.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
        ax.set_xticks(range(display.shape[1]))
        ax.set_xticklabels(display.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(display.shape[0]))
        ax.set_yticklabels(display.index, fontsize=8)
        ax.set_title(f"h = {h}")

        # Annotate cells with significance stars
        for i in range(display.shape[0]):
            for j in range(display.shape[1]):
                pval = pivot.iloc[i, j]
                if np.isnan(pval):
                    text = ""
                elif pval < 0.01:
                    text = "***"
                elif pval < 0.05:
                    text = "**"
                elif pval < 0.10:
                    text = "*"
                else:
                    text = ""
                ax.text(j, i, text, ha="center", va="center", fontsize=10, color="black")

    fig.colorbar(im, ax=axes, label=r"$-\log_{10}(p)$", shrink=0.8)
    fig.suptitle(r"Robustness: $\gamma_h$ (Shock $\times$ VolProxy) significance", y=1.02)
    fig.tight_layout()
    return fig
