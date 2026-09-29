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


class QuantileLSTMForecaster(LSTMForecaster):
    """Same LSTM body; the head predicts several quantiles of scaled glucose at the horizon.

    Output (B, K), ordered like ``quantiles`` (ascending). Crossing is impossible by construction:
    the head predicts the median directly and, for the other quantiles, non-negative gaps
    (softplus) that are accumulated upward from the median for tau > 0.5 and downward for
    tau < 0.5. Scaling to mg/dL is affine with a positive factor, so the order survives unscaling."""

    def __init__(self, n_features: int, quantiles: list[float], hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__(n_features, hidden_size, num_layers, dropout)
        q = [float(t) for t in quantiles]
        if q != sorted(q) or len(set(q)) != len(q) or 0.5 not in q:
            raise ValueError("quantiles must be strictly increasing and include 0.5")
        self.quantiles = q
        self.median_index = q.index(0.5)
        self.head = nn.Linear(hidden_size, len(q))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = super().forward(x)
        m = self.median_index
        median = raw[:, m : m + 1]
        gaps = nn.functional.softplus(raw)
        up = median + torch.cumsum(gaps[:, m + 1 :], dim=1)
        down = median - torch.cumsum(gaps[:, :m].flip(1), dim=1).flip(1)
        return torch.cat([down, median, up], dim=1)


class LSTMClassifier(LSTMForecaster):
    """Same LSTM body; the head outputs one logit for P(glucose < 70 mg/dL at the horizon)."""
