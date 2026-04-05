"""hangar.ml.volatility.training — Training loop and evaluation for vol forecasters."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from hangar.ml.checkpoints import CheckpointMeta, save_checkpoint
from hangar.ml.volatility.dataset import VolatilityDataset, create_walk_forward_splits


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    device: str = "cpu"
    checkpoint_every: int = 0
    model_name: str = "vol_forecaster"


@dataclass
class TrainResult:
    """Container for training outputs."""

    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    best_val_loss: float
    checkpoint_path: Optional[str] = None


def train_model(
    model: nn.Module,
    dataset: VolatilityDataset,
    *,
    train_indices: range,
    val_indices: range,
    config: Optional[TrainConfig] = None,
) -> TrainResult:
    """Train a volatility forecasting model on a single train/val split.

    Parameters
    ----------
    model
        PyTorch model (LSTMVolModel or GRUVolModel).
    dataset
        The full VolatilityDataset.
    train_indices
        Index range for training samples.
    val_indices
        Index range for validation samples.
    config
        Training hyperparameters.

    Returns
    -------
    TrainResult
        Training history, best epoch, and checkpoint path.
    """
    if config is None:
        config = TrainConfig()

    device = torch.device(config.device)
    model = model.to(device)

    train_subset = Subset(dataset, list(train_indices))
    val_subset = Subset(dataset, list(val_indices))

    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    # Restore best weights
    checkpoint_path = None
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        meta = CheckpointMeta(
            model_name=config.model_name,
            model_class=type(model).__qualname__,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            epoch=best_epoch,
            metrics={"val_mse": best_val_loss, "train_mse": train_losses[best_epoch]},
            hyperparameters={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        )
        pt_path = save_checkpoint(model, meta)
        checkpoint_path = str(pt_path)

    return TrainResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        checkpoint_path=checkpoint_path,
    )


def walk_forward_train(
    model_factory: Callable[[], nn.Module],
    dataset: VolatilityDataset,
    *,
    config: Optional[TrainConfig] = None,
    min_train_size: int = 504,
    val_size: int = 63,
    step_size: int = 63,
    expanding: bool = True,
) -> List[TrainResult]:
    """Run walk-forward training across multiple temporal splits.

    Parameters
    ----------
    model_factory
        Callable that returns a fresh model instance (called once per fold).
    dataset
        The full VolatilityDataset.
    config
        Training hyperparameters (shared across folds).
    min_train_size
        Minimum training set size for the first fold.
    val_size
        Validation set size per fold.
    step_size
        Step between folds.
    expanding
        Expanding vs rolling training window.

    Returns
    -------
    List[TrainResult]
        One TrainResult per walk-forward fold.
    """
    splits = create_walk_forward_splits(
        len(dataset),
        min_train_size=min_train_size,
        val_size=val_size,
        step_size=step_size,
        expanding=expanding,
    )

    results: List[TrainResult] = []
    for train_idx, val_idx in splits:
        model = model_factory()
        result = train_model(
            model, dataset, train_indices=train_idx, val_indices=val_idx, config=config
        )
        results.append(result)

    return results


def evaluate_model(
    model: nn.Module,
    dataset: VolatilityDataset,
    indices: range,
    *,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a trained model on a data subset.

    Parameters
    ----------
    model
        Trained PyTorch model.
    dataset
        The VolatilityDataset.
    indices
        Index range for evaluation.
    device
        Device for inference.

    Returns
    -------
    Dict[str, float]
        Metrics: ``mse``, ``mae``, ``rmse``, ``directional_accuracy``.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=256, shuffle=False)

    preds: List[torch.Tensor] = []
    actuals: List[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(dev)
            pred = model(x_batch).cpu()
            preds.append(pred)
            actuals.append(y_batch)

    pred_arr = torch.cat(preds).numpy().flatten()
    actual_arr = torch.cat(actuals).numpy().flatten()

    errors = pred_arr - actual_arr
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    # Directional accuracy: did predicted vol change direction match actual?
    if len(pred_arr) > 1:
        pred_diff = np.diff(pred_arr)
        actual_diff = np.diff(actual_arr)
        direction_match = np.sign(pred_diff) == np.sign(actual_diff)
        directional_accuracy = float(np.mean(direction_match))
    else:
        directional_accuracy = float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def predict_series(
    model: nn.Module,
    dataset: VolatilityDataset,
    *,
    device: str = "cpu",
) -> pd.Series:
    """Generate a full prediction series (sigma_hat) from a trained model.

    Parameters
    ----------
    model
        Trained volatility forecaster.
    dataset
        VolatilityDataset covering the desired prediction range.
    device
        Device for inference.

    Returns
    -------
    pd.Series
        Predicted volatility series indexed by date, named ``"sigma_hat"``.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    preds: List[torch.Tensor] = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(dev)
            pred = model(x_batch).cpu()
            preds.append(pred)

    pred_arr = torch.cat(preds).numpy().flatten()
    return pd.Series(pred_arr, index=dataset.dates[: len(pred_arr)], name="sigma_hat")
