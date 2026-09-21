"""Shared helpers: config loading and logging."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML config into a dict. A ``base: <file>`` key (path relative to this config)
    is loaded first and the file's own keys are deep-merged over it."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    return _deep_merge(load_config(path.parent / base), cfg) if base else cfg


def get_logger(name: str) -> logging.Logger:
    """Module logger with a single stream handler (idempotent, no import-time side effects)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) for reproducible runs."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(pred, target) -> float:
    """Root mean squared error; inputs in mg/dL, result in mg/dL."""
    import numpy as np

    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(d**2)))


def mae(pred, target) -> float:
    """Mean absolute error; inputs in mg/dL, result in mg/dL."""
    import numpy as np

    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(d)))


def per_patient_metrics(pred, target, patients):
    """RMSE / MAE (mg/dL) per patient. Metrics are computed within each patient's own windows,
    then averaged across patients by ``summarize_patients`` (not pooled)."""
    import numpy as np
    import pandas as pd

    pred, target, patients = np.asarray(pred), np.asarray(target), np.asarray(patients)
    rows = []
    for p in sorted(set(patients.tolist())):
        m = patients == p
        rows.append({"patient": p, "n_windows": int(m.sum()), "rmse": rmse(pred[m], target[m]), "mae": mae(pred[m], target[m])})
    return pd.DataFrame(rows)


def summarize_patients(table):
    """Mean and sample std (ddof=1) across patients of per-patient RMSE and MAE."""
    return {
        "rmse_mean": table["rmse"].mean(),
        "rmse_std": table["rmse"].std(ddof=1),
        "mae_mean": table["mae"].mean(),
        "mae_std": table["mae"].std(ddof=1),
    }


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
