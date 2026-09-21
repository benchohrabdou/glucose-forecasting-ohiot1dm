"""Model registry: every model maps (B, window_len, n_features) -> (B, 1) scaled glucose."""
import torch


def build_model(cfg: dict, n_features: int) -> torch.nn.Module:
    """Instantiate the network named by ``cfg['model']['type']``."""
    spec = dict(cfg["model"])
    kind = spec.pop("type")
    if kind == "lstm":
        from src.models.lstm import LSTMForecaster

        return LSTMForecaster(n_features, **spec)
    raise ValueError(f"unknown model type: {kind!r}")
