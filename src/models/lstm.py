"""LSTM forecaster: (B, window_len, n_features) -> (B, 1), in scaled glucose space."""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """1-2 layer LSTM; final hidden state of the top layer -> linear head.

    ``dropout`` is applied between stacked layers (PyTorch ignores it for one layer) and on the
    final hidden state before the head, so a single-layer model is regularised too."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1]))
