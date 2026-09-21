"""Baselines sharing the model interface: input (B, window_len, n_features) -> output (B, 1),
both in SCALED glucose space (the caller unscales to mg/dL before any metric).

Column 0 of x is always scaled glucose (see ``feature_columns``). Forward-filled bins repeat the
last real reading, so "last value" is the last observed glucose within the 30-min fill limit.

CLI: ``python -m src.models.baselines --config configs/base.yaml`` runs every baseline at both
horizons, chooses each baseline's only hyperparameter on VALIDATION, scores TEST, and writes
tables to results/.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from src.data.dataset import build_datasets, collect_arrays
from src.evaluate import write_comparison, write_coverage, write_fingerprint
from src.utils import get_logger, load_config, per_patient_metrics, set_seed

log = get_logger(__name__)

HORIZONS = (6, 12)  # steps of 5 min = 30 and 60 min
K_CANDIDATES = (2, 3, 4, 6)  # linear-extrapolation fit lengths, chosen on validation
ALPHA_CANDIDATES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)  # ridge penalties, chosen on validation


class Persistence:
    """Predict the last glucose value in the window."""

    name = "persistence"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -1, 0:1]


class LinearExtrapolation:
    """Least-squares slope over the last ``k`` glucose values, extrapolated ``horizon`` steps
    from the last value. Scaling is affine, so extrapolating in scaled space is equivalent to
    extrapolating in mg/dL."""

    def __init__(self, horizon: int, k: int = 3) -> None:
        self.horizon, self.k = horizon, k
        self.name = f"linear_k{k}"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x[:, -self.k :, 0]
        t = torch.arange(self.k, dtype=y.dtype) - (self.k - 1) / 2
        slope = (t * (y - y.mean(dim=1, keepdim=True))).sum(dim=1) / (t**2).sum()
        return (y[:, -1] + slope * self.horizon).unsqueeze(1)


class RidgeRegression:
    """Ridge on the flattened window (all features, scaled). Fit on training windows only."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.name = "ridge"
        self._model = Ridge(alpha=alpha)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "RidgeRegression":
        self._model.fit(x.reshape(len(x), -1).numpy(), y.numpy().ravel())
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pred = self._model.predict(x.reshape(len(x), -1).numpy())
        return torch.from_numpy(pred.astype(np.float32)).unsqueeze(1)


def _rmse_mgdl(model, x, y, scaler) -> float:
    err = scaler.unscale_glucose(model(x).numpy()) - scaler.unscale_glucose(y.numpy())
    return float(np.sqrt(np.mean(err**2)))


def run_horizon(cfg: dict, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All baselines at one horizon -> (per-patient TEST rows, validation-selection rows)."""
    cfg = copy.deepcopy(cfg)
    cfg["window"]["horizon"] = horizon
    train, val, test, scaler = build_datasets(cfg)
    xtr, ytr, _, _ = collect_arrays(train)
    xva, yva, _, _ = collect_arrays(val)
    xte, yte, pte, tte = collect_arrays(test)
    minutes = horizon * cfg["grid"]["step_min"]
    write_fingerprint(cfg, minutes, pte, tte)  # lets every later model prove it uses these exact windows

    # Hyperparameters are chosen on validation only; test is touched once, afterwards.
    val_rows = []
    for k in K_CANDIDATES:
        val_rows.append({"horizon_min": minutes, "model": "linear", "param": f"k={k}", "val_rmse": _rmse_mgdl(LinearExtrapolation(horizon, k), xva, yva, scaler)})
    ridge_fits = {a: RidgeRegression(a).fit(xtr, ytr) for a in ALPHA_CANDIDATES}
    for a, m in ridge_fits.items():
        val_rows.append({"horizon_min": minutes, "model": "ridge", "param": f"alpha={a}", "val_rmse": _rmse_mgdl(m, xva, yva, scaler)})
    val_rows.append({"horizon_min": minutes, "model": "persistence", "param": "", "val_rmse": _rmse_mgdl(Persistence(), xva, yva, scaler)})
    val_df = pd.DataFrame(val_rows)

    best_k = min(K_CANDIDATES, key=lambda k: _rmse_mgdl(LinearExtrapolation(horizon, k), xva, yva, scaler))
    best_alpha = min(ALPHA_CANDIDATES, key=lambda a: _rmse_mgdl(ridge_fits[a], xva, yva, scaler))
    log.info("PH=%d min: linear k=%d, ridge alpha=%g (chosen on validation)", minutes, best_k, best_alpha)

    y_mgdl = scaler.unscale_glucose(yte.numpy()).ravel()
    rows = []
    for model, label in (
        (Persistence(), "persistence"),
        (LinearExtrapolation(horizon, best_k), f"linear (k={best_k})"),
        (ridge_fits[best_alpha], f"ridge (alpha={best_alpha:g})"),
    ):
        pred = scaler.unscale_glucose(model(xte).numpy()).ravel()
        table = per_patient_metrics(pred, y_mgdl, pte)
        table.insert(0, "model", label)
        table.insert(1, "horizon_min", minutes)
        table["seed"] = cfg["seed"]
        rows.append(table)
    return pd.concat(rows, ignore_index=True), val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines at both horizons.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    per_patient, validation = [], []
    for h in HORIZONS:
        p, v = run_horizon(cfg, h)
        per_patient.append(p), validation.append(v)
    per_patient, validation = pd.concat(per_patient, ignore_index=True), pd.concat(validation, ignore_index=True)

    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    per_patient.to_csv(out / "baselines_per_patient.csv", index=False)
    validation.to_csv(out / "baselines_validation.csv", index=False)
    write_coverage(cfg, HORIZONS)
    summary = write_comparison(cfg)
    log.info("wrote baselines_per_patient, baselines_validation, test_coverage, comparison_* to %s", out)
    print(summary[["cohort", "horizon_min", "model", "n_patients", "rmse_mean", "rmse_std"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
