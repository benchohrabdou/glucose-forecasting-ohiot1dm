"""Evaluation and reporting.

Shared helpers (used by the baselines runner and by checkpoint evaluation):
  * ``write_comparison``  - per-patient and cohort-level tables (all / 2018 / 2020) from every
    ``baselines_per_patient.csv`` and ``model_*_per_patient.csv`` in results/;
  * ``write_coverage``    - test CGM readings scored vs. total, per patient and horizon;
  * ``write_fingerprint`` / ``verify_fingerprint`` - proof that two runs are scored on exactly
    the same test windows.

CLI: ``python -m src.evaluate --config configs/lstm_ph30.yaml --checkpoint checkpoints/lstm_ph30.pt``
All metrics are RMSE / MAE in mg/dL, computed per patient on that patient's test file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    Scaler,
    build_datasets,
    collect_arrays,
    load_grids,
    make_segments,
    valid_window_ends,
    window_fingerprint,
)
from src.models import build_model
from src.utils import get_logger, load_config, per_patient_metrics, summarize_patients

log = get_logger(__name__)

FINGERPRINT_FILE = "test_windows_fingerprint.csv"


def _results_dir(cfg: dict) -> Path:
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out


def cohort_of(cfg: dict) -> dict[int, str]:
    """patient id -> cohort year ('2018' / '2020')."""
    return {p: str(year) for year, ps in cfg["cohorts"].items() for p in ps}


def write_comparison(cfg: dict) -> pd.DataFrame:
    """Combine every per-patient result file into ``comparison_per_patient.csv`` and
    ``comparison_summary.csv`` (mean +/- std across patients, for all patients and per cohort).
    The published-benchmark column is intentionally left empty for the project owner to fill in."""
    out = _results_dir(cfg)
    files = [out / "baselines_per_patient.csv", *sorted(out.glob("model_*_per_patient.csv"))]
    per_patient = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    per_patient["cohort"] = per_patient["patient"].map(cohort_of(cfg))
    per_patient.to_csv(out / "comparison_per_patient.csv", index=False)

    rows = []
    for cohort in ("all", *sorted(set(cohort_of(cfg).values()))):
        sub = per_patient if cohort == "all" else per_patient[per_patient["cohort"] == cohort]
        for (model, horizon), g in sub.groupby(["model", "horizon_min"], sort=False):
            rows.append({"cohort": cohort, "model": model, "horizon_min": horizon, "n_patients": len(g),
                         **summarize_patients(g), "published_bglp_rmse": ""})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "comparison_summary.csv", index=False)
    return summary


def write_coverage(cfg: dict, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Per test file and horizon: real CGM readings, how many are scored, and why the rest are not.

    unscored_no_history: the reading is too early in the file to have a full input window
    (first window_len - 1 + horizon bins); unscored_input_gap: its input window has a gap the
    fill rules do not cover (NaN input or too many imputed bins). Targets that are not real
    readings are not readings, so they never appear here."""
    w = cfg["window"]
    patients = [p for ps in cfg["cohorts"].values() for p in ps]
    segments = make_segments(cfg, load_grids(cfg["paths"]["processed_dir"], patients, "test"), "test")
    cohorts = cohort_of(cfg)
    rows = []
    for horizon in horizons:
        for pid, grid in segments:
            glucose = grid["glucose"].to_numpy(dtype=np.float64)
            observed = grid["glucose_observed"].to_numpy(dtype=bool)
            _, c = valid_window_ends(glucose, observed, w["window_len"], horizon, w["max_imputed_frac"])
            total = int(observed.sum())
            with_history = c["candidates"] - c["rejected_target_not_observed"]
            row = {
                "patient": pid, "cohort": cohorts[pid], "horizon_min": horizon * cfg["grid"]["step_min"],
                "test_cgm_readings": total, "scored": c["kept"],
                "pct_scored": round(100 * c["kept"] / total, 1),
                "unscored_no_history": total - with_history,
                "unscored_input_gap": c["rejected_nan_input"] + c["rejected_imputed_frac"],
            }
            assert row["scored"] + row["unscored_no_history"] + row["unscored_input_gap"] == total
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(_results_dir(cfg) / "test_coverage.csv", index=False)
    return table


def write_fingerprint(cfg: dict, horizon_min: int, patients: np.ndarray, target_ts: np.ndarray) -> None:
    """Record the test-window fingerprint of a run (upsert by horizon and window length)."""
    path = _results_dir(cfg) / FINGERPRINT_FILE
    row = {"horizon_min": horizon_min, "window_len": cfg["window"]["window_len"],
           "n_windows": len(patients), "sha256": window_fingerprint(patients, target_ts)}
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    old = old[~((old["horizon_min"] == horizon_min) & (old["window_len"] == row["window_len"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def verify_fingerprint(cfg: dict, horizon_min: int, patients: np.ndarray, target_ts: np.ndarray) -> None:
    """Raise unless the test windows are identical to the ones the baselines were scored on."""
    path = Path(cfg["paths"]["results_dir"]) / FINGERPRINT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m src.models.baselines` first")
    ref = pd.read_csv(path)
    ref = ref[(ref["horizon_min"] == horizon_min) & (ref["window_len"] == cfg["window"]["window_len"])]
    if ref.empty:
        raise ValueError(f"no baseline scored PH={horizon_min} min with window_len={cfg['window']['window_len']}")
    got = window_fingerprint(patients, target_ts)
    if got != ref["sha256"].item():
        raise AssertionError("test windows differ from the baselines' windows; comparison would not be like-for-like")


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Batched forward pass in eval mode; returns scaled predictions (N, 1)."""
    was_training = model.training
    model.eval()
    out = torch.cat([model(x[i : i + batch_size]) for i in range(0, len(x), batch_size)])
    model.train(was_training)
    return out


def model_label(cfg: dict) -> str:
    return f"{cfg['model']['type']} ({'glucose + insulin/carbs' if cfg['features']['insulin_carbs'] else 'glucose only'})"


def evaluate_checkpoint(cfg: dict, checkpoint: str | Path, name: str) -> pd.DataFrame:
    """Score a checkpoint on the test files with the scaler saved alongside it."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("window", "features", "model", "gap", "split"):
        if cfg[key] != ckpt["cfg"][key]:
            raise ValueError(f"config section '{key}' differs from the one the checkpoint was trained with")
    scaler = Scaler(**ckpt["scaler"])  # never refit at evaluation time
    _, _, test, _ = build_datasets(cfg, scaler=scaler)
    x, y, patients, target_ts = collect_arrays(test)
    horizon_min = cfg["window"]["horizon"] * cfg["grid"]["step_min"]
    verify_fingerprint(cfg, horizon_min, patients, target_ts)

    model = build_model(cfg, x.shape[2])
    model.load_state_dict(ckpt["model_state"])
    pred = scaler.unscale_glucose(predict(model, x).numpy()).ravel()
    table = per_patient_metrics(pred, scaler.unscale_glucose(y.numpy()).ravel(), patients)
    table.insert(0, "model", model_label(cfg))
    table.insert(1, "horizon_min", horizon_min)
    table["seed"] = ckpt["seed"]
    table.to_csv(_results_dir(cfg) / f"model_{name}_per_patient.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_checkpoint(cfg, args.checkpoint, Path(args.config).stem)
    summary = write_comparison(cfg)
    h = cfg["window"]["horizon"] * cfg["grid"]["step_min"]
    shown = summary[summary["horizon_min"] == h].copy()
    shown["RMSE"] = shown.rmse_mean.round(2).astype(str) + " ± " + shown.rmse_std.round(2).astype(str)
    shown["MAE"] = shown.mae_mean.round(2).astype(str) + " ± " + shown.mae_std.round(2).astype(str)
    print(shown[["cohort", "model", "n_patients", "RMSE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
