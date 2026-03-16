"""
harbor.ml.volatility — Volatility forecasters (neural network + classical baselines).

Provides LSTM/GRU-based volatility forecasting with walk-forward training,
classical baselines (GARCH(1,1), EWMA, rolling historical) for benchmarking,
and integration into vol-targeting and risk-parity strategies.

Status: NN models are experimental scaffolding pending H3 validation.
Classical baselines (GARCH, EWMA) are implemented and ready for comparison.
"""

from harbor.ml.volatility.baselines import (
    BaselineResult,
    evaluate_forecast,
    ewma_volatility,
    fit_garch11,
    garch11_forecast,
    rolling_volatility,
    run_baseline_comparison,
)
from harbor.ml.volatility.dataset import (
    VolatilityDataset,
    VolDatasetConfig,
    build_features,
    build_target,
    create_walk_forward_splits,
)
from harbor.ml.volatility.integration import (
    sigma_hat_to_regime_proxy,
    vol_scaled_weight_func,
)
from harbor.ml.volatility.models import (
    GRUVolModel,
    LSTMVolModel,
    create_model,
)
from harbor.ml.volatility.training import (
    TrainConfig,
    TrainResult,
    evaluate_model,
    predict_series,
    train_model,
    walk_forward_train,
)

__all__ = [
    "BaselineResult",
    "GRUVolModel",
    "LSTMVolModel",
    "TrainConfig",
    "TrainResult",
    "VolDatasetConfig",
    "VolatilityDataset",
    "build_features",
    "build_target",
    "create_model",
    "create_walk_forward_splits",
    "evaluate_forecast",
    "evaluate_model",
    "ewma_volatility",
    "fit_garch11",
    "garch11_forecast",
    "predict_series",
    "rolling_volatility",
    "run_baseline_comparison",
    "sigma_hat_to_regime_proxy",
    "train_model",
    "vol_scaled_weight_func",
    "walk_forward_train",
]
