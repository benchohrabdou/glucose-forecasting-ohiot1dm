"""Model registry. ``lstm``: (B, window_len, n_features) -> (B, 1) scaled glucose;
``lstm_quantile``: -> (B, K) scaled glucose quantiles; ``lstm_classifier``: -> (B, 1) logit of glucose < 70."""
import torch


def build_model(cfg: dict, n_features: int) -> torch.nn.Module:
    """Instantiate the network named by ``cfg['model']['type']``."""
    spec = dict(cfg["model"])
    kind = spec.pop("type")
    if kind == "lstm":
        from src.models.lstm import LSTMForecaster

        return LSTMForecaster(n_features, **spec)
    if kind == "lstm_quantile":
        from src.models.lstm import QuantileLSTMForecaster

        return QuantileLSTMForecaster(n_features, **spec)
    if kind == "lstm_classifier":
        from src.models.lstm import LSTMClassifier

        return LSTMClassifier(n_features, **spec)
    raise ValueError(f"unknown model type: {kind!r}")
