"""Vol forecasting training orchestrator.

Pipeline:
1) Load prices, compute equal-weighted market return
2) Create VolatilityDataset with config
3) For each architecture: walk_forward_train -> find best fold -> predict_series
   -> sigma_hat_to_regime_proxy
4) Save: sigma_hat CSV, regime_proxy CSV, fold_metrics JSON per architecture
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.data import load_sp500_prices, load_sp500_tickers  # noqa: E402
from harbor.ml.checkpoints import load_checkpoint  # noqa: E402
from harbor.ml.volatility import (  # noqa: E402
    TrainConfig,
    VolatilityDataset,
    VolDatasetConfig,
    create_model,
    predict_series,
    sigma_hat_to_regime_proxy,
    walk_forward_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train volatility forecasting models.")
    parser.add_argument("--start", default="2010-01-01", help="Start date")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="End date",
    )
    parser.add_argument(
        "--architecture",
        choices=["lstm", "gru", "both"],
        default="both",
        help="Model architecture(s) to train",
    )
    parser.add_argument("--device", default="cpu", help="Training device")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per fold")
    parser.add_argument("--output-dir", default="output/vol_forecast", help="Output directory")
    parser.add_argument("--seq-len", type=int, default=60, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden state dimension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print("[1/4] Loading prices...")
    tickers = load_sp500_tickers(as_of=args.end)[:50]
    prices = load_sp500_prices(tickers=tickers, start=args.start, end=args.end, adjusted=True)
    returns = prices.pct_change().dropna(how="all")
    min_obs = max(int(0.8 * len(returns)), 252)
    returns = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)

    # Equal-weighted market return
    market_return = returns.mean(axis=1)
    market_return.name = "market"
    print(f"  Market return series: {len(market_return)} days")

    # 2) Create dataset
    print("[2/4] Building dataset...")
    ds_config = VolDatasetConfig(seq_len=args.seq_len)
    dataset = VolatilityDataset(market_return, config=ds_config)
    n_features = dataset.n_features
    print(f"  Dataset: {len(dataset)} samples, {n_features} features")

    # Determine architectures
    archs = ["lstm", "gru"] if args.architecture == "both" else [args.architecture]

    # 3) Train each architecture
    for arch in archs:
        print(f"\n[3/4] Training {arch.upper()}...")

        def model_factory(a=arch):
            return create_model(
                a,
                n_features,
                hidden_size=args.hidden_size,
            )

        fold_results = walk_forward_train(
            model_factory,
            dataset,
            config=TrainConfig(
                epochs=args.epochs,
                device=args.device,
                model_name=f"vol_{arch}",
            ),
        )

        # Find best fold
        best_idx = min(range(len(fold_results)), key=lambda i: fold_results[i].best_val_loss)
        best = fold_results[best_idx]
        print(
            f"  Best fold: {best_idx} (val_loss={best.best_val_loss:.6f}, "
            f"epoch={best.best_epoch})"
        )

        # Load best checkpoint and predict
        if best.checkpoint_path:
            model = model_factory()
            load_checkpoint(model, Path(best.checkpoint_path), device=args.device)

            sigma_hat = predict_series(model, dataset, device=args.device)
            regime_proxy = sigma_hat_to_regime_proxy(sigma_hat)

            # Save outputs
            sigma_hat.to_csv(out / f"sigma_hat_{arch}.csv", header=True)
            regime_proxy.to_csv(out / f"regime_proxy_{arch}.csv", header=True)
            print(f"  Saved sigma_hat and regime_proxy for {arch}")

        # Save fold metrics
        fold_metrics = []
        for i, fr in enumerate(fold_results):
            fold_metrics.append(
                {
                    "fold": i,
                    "best_epoch": fr.best_epoch,
                    "best_val_loss": fr.best_val_loss,
                    "checkpoint_path": fr.checkpoint_path,
                    "n_train_epochs": len(fr.train_losses),
                }
            )
        with open(out / f"fold_metrics_{arch}.json", "w") as f:
            json.dump(fold_metrics, f, indent=2)

    print("\n[4/4] Done.")


if __name__ == "__main__":
    main()
