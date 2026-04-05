"""hangar.ml.volatility.models — LSTM/GRU volatility forecasting architectures."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class LSTMVolModel(nn.Module):
    """LSTM-based volatility forecaster.

    Architecture::

        Input (batch, seq_len, n_features)
        -> LSTM(n_features, hidden_size, num_layers, dropout)
        -> Linear(hidden_size, 1)
        -> ReLU (volatility is non-negative)

    Parameters
    ----------
    n_features
        Number of input features per timestep.
    hidden_size
        LSTM hidden state dimension.
    num_layers
        Number of stacked LSTM layers.
    dropout
        Dropout between LSTM layers (applied only if num_layers > 1).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        torch.Tensor
            Predicted volatility of shape ``(batch, 1)``.
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        return self.relu(self.fc(last_hidden))


class GRUVolModel(nn.Module):
    """GRU-based volatility forecaster.

    Same interface and output as :class:`LSTMVolModel`, but uses GRU cells.

    Parameters
    ----------
    n_features
        Number of input features per timestep.
    hidden_size
        GRU hidden state dimension.
    num_layers
        Number of stacked GRU layers.
    dropout
        Dropout between GRU layers.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns
        -------
        torch.Tensor
            Predicted volatility of shape ``(batch, 1)``.
        """
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.relu(self.fc(last_hidden))


def create_model(
    architecture: Literal["lstm", "gru"],
    n_features: int,
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory function to create a volatility forecasting model.

    Parameters
    ----------
    architecture
        Model type: ``"lstm"`` or ``"gru"``.
    n_features
        Number of input features per timestep.
    hidden_size
        Hidden state dimension.
    num_layers
        Number of recurrent layers.
    dropout
        Dropout rate.

    Returns
    -------
    nn.Module
        An instance of :class:`LSTMVolModel` or :class:`GRUVolModel`.

    Raises
    ------
    ValueError
        If ``architecture`` is not ``"lstm"`` or ``"gru"``.
    """
    kwargs = dict(
        n_features=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    if architecture == "lstm":
        return LSTMVolModel(**kwargs)
    if architecture == "gru":
        return GRUVolModel(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture!r}. Use 'lstm' or 'gru'.")
