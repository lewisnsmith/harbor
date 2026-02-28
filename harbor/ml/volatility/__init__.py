"""
harbor.ml.volatility — Neural network volatility forecasters.

Status: Experimental scaffolding. Implemented ahead of the H3 roadmap as
exploratory work. Unit tests pass but models have not been validated against
classical baselines (GARCH, EWMA) on out-of-sample data.

Provides LSTM/GRU-based volatility forecasting with walk-forward training
and integration into vol-targeting and risk-parity strategies.
"""

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
    "evaluate_model",
    "predict_series",
    "sigma_hat_to_regime_proxy",
    "train_model",
    "vol_scaled_weight_func",
    "walk_forward_train",
]
