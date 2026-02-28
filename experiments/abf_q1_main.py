"""ABF Q1 experiment orchestrator — Shock -> Persistence -> Reversal.

Pipeline:
1) Load universe and prices
2) Compute returns and market return
3) Detect shocks and build vol proxy
4) Build control matrix
5) Run local projections at h=1, 5, 21
6) Compute CARs at each horizon
7) Robustness sweep (unless --skip-robustness)
8) Generate figures
9) Save outputs (coefficients JSON, sweep CSV, CAR CSV)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow direct script execution without pip install -e .
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.abf.q1.analysis import (  # noqa: E402
    build_control_matrix,
    fit_local_projections,
)
from harbor.abf.q1.robustness import robustness_sweep  # noqa: E402
from harbor.abf.q1.visualization import (  # noqa: E402
    plot_car_paths,
    plot_coefficient_path,
    plot_robustness_heatmap,
    plot_shock_timeline,
)
from harbor.backtest.metrics import cumulative_abnormal_return  # noqa: E402
from harbor.data import load_sp500_prices, load_sp500_tickers  # noqa: E402
from harbor.risk.regime_detection import (  # noqa: E402
    detect_vol_shocks,
    vol_control_pressure_proxy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ABF Q1 Shock -> Persistence -> Reversal.")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="End date",
    )
    parser.add_argument("--max-assets", type=int, default=75, help="Max constituents")
    parser.add_argument("--output-dir", default="output/abf_q1", help="Output directory")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load tickers + prices
    print("[1/9] Loading universe and prices...")
    tickers = load_sp500_tickers(as_of=args.end)[: args.max_assets]
    prices = load_sp500_prices(tickers=tickers, start=args.start, end=args.end, adjusted=True)
    returns = prices.pct_change().dropna(how="all")

    # Keep liquid/continuous names
    min_obs = max(int(0.8 * len(returns)), 252)
    returns = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)
    print(f"  Universe: {returns.shape[1]} assets, {returns.shape[0]} days")

    # 2) Compute equal-weighted market return
    print("[2/9] Computing market return...")
    market_return = returns.mean(axis=1)

    # 3) Detect shocks + vol proxy
    print("[3/9] Detecting vol shocks...")
    shocks = detect_vol_shocks(market_return, threshold_pct=0.95)
    vol_proxy = vol_control_pressure_proxy(market_return)
    print(f"  Shock days: {shocks.sum()} out of {len(shocks)}")

    # 4) Build control matrix
    print("[4/9] Building control matrix...")
    controls = build_control_matrix(market_return, market_returns=market_return)

    # 5) Local projections
    print("[5/9] Running local projections (h=1, 5, 21)...")
    horizons = [1, 5, 21]
    lp_results = fit_local_projections(
        market_return, shocks, vol_proxy, controls, horizons=horizons
    )
    for h, res in lp_results.items():
        b_h = res.params.get("shock", float("nan"))
        c_h = res.params.get("shock_x_vol_proxy", float("nan"))
        print(f"  h={h:>2}: b_h={b_h:+.6f}  c_h={c_h:+.6f}  R²={res.rsquared:.4f}")

    # 6) Cumulative abnormal returns
    print("[6/9] Computing CARs...")
    shock_dates = shocks.index[shocks]
    car_results = {}
    for h in horizons:
        car_df = cumulative_abnormal_return(market_return, shock_dates, horizon=h)
        car_results[h] = car_df
        if not car_df.empty:
            print(f"  h={h:>2}: mean CAR = {car_df['car'].mean():+.6f} (n={len(car_df)})")

    # 7) Robustness sweep
    sweep_results = None
    if not args.skip_robustness:
        print("[7/9] Running robustness sweep...")
        config_path = REPO_ROOT / "configs" / "abf" / "q1_shock_definitions.json"
        with open(config_path) as f:
            shock_cfg = json.load(f)

        all_configs = [shock_cfg["primary_shock_definition"]]
        all_configs.extend(shock_cfg.get("alternative_shock_definitions", []))
        splits = ["full", "pre_2020", "post_2020"]

        sweep_results = robustness_sweep(
            market_return, all_configs, splits, horizons, vol_proxy, controls
        )
        print(f"  Sweep: {len(sweep_results)} rows")
    else:
        print("[7/9] Robustness sweep skipped.")

    # 8) Generate figures
    print("[8/9] Generating figures...")

    fig_timeline = plot_shock_timeline(market_return, shocks, vol_proxy)
    fig_timeline.savefig(fig_dir / "shock_timeline.png", dpi=150, bbox_inches="tight")

    fig_car = plot_car_paths(car_results, horizons)
    fig_car.savefig(fig_dir / "car_paths.png", dpi=150, bbox_inches="tight")

    fig_coeff = plot_coefficient_path(lp_results, horizons)
    fig_coeff.savefig(fig_dir / "coefficient_path.png", dpi=150, bbox_inches="tight")

    if sweep_results is not None and not sweep_results.empty:
        fig_heat = plot_robustness_heatmap(sweep_results)
        fig_heat.savefig(fig_dir / "robustness_heatmap.png", dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close("all")
    print(f"  Saved to {fig_dir}")

    # 9) Save data outputs
    print("[9/9] Saving data outputs...")

    # Coefficients JSON
    coeff_data = {}
    for h, res in lp_results.items():
        coeff_data[str(h)] = {
            "params": res.params.to_dict(),
            "bse": res.bse.to_dict(),
            "pvalues": res.pvalues.to_dict(),
            "rsquared": res.rsquared,
            "nobs": int(res.nobs),
        }
    with open(out / "coefficients.json", "w") as f:
        json.dump(coeff_data, f, indent=2)

    # CAR tables
    for h, car_df in car_results.items():
        if not car_df.empty:
            car_df.to_csv(out / f"car_h{h}.csv")

    # Sweep CSV
    if sweep_results is not None:
        sweep_results.to_csv(out / "robustness_sweep.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()
