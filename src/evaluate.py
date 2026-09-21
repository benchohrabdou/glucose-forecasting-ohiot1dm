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


GLUCOSE_LABEL = "lstm (glucose only)"
INSULIN_LABEL = "lstm (glucose + insulin/carbs)"


def _model_order(model: str) -> int:
    for rank, prefix in enumerate(("persistence", "linear", "ridge", GLUCOSE_LABEL, INSULIN_LABEL)):
        if model.startswith(prefix):
            return rank
    return 9


def _load_runs(cfg: dict) -> pd.DataFrame:
    """Per-patient rows of the baselines and of every multi-seed model run
    (``model_*_seed*_per_patient.csv``), tagged with the patient's cohort."""
    out = _results_dir(cfg)
    files = [out / "baselines_per_patient.csv", *sorted(out.glob("model_*_seed*_per_patient.csv"))]
    runs = pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)
    runs["cohort"] = runs["patient"].map(cohort_of(cfg))
    return runs


def _cohorts(cfg: dict) -> list[str]:
    return ["all", *sorted(set(cohort_of(cfg).values()))]


def _subset(runs: pd.DataFrame, cohort: str) -> pd.DataFrame:
    return runs if cohort == "all" else runs[runs["cohort"] == cohort]


def write_comparison(cfg: dict) -> pd.DataFrame:
    """Combine baselines and multi-seed model runs into three tables.

    * ``comparison_per_patient.csv``: RMSE / MAE per patient, AVERAGED OVER SEEDS.
    * ``comparison_summary.csv``: per cohort (all / 2018 / 2020), model and horizon:
      ``rmse_mean`` +/- ``rmse_std`` across patients of the seed-averaged values, and, for models
      run with several seeds, ``rmse_seed_mean`` +/- ``rmse_seed_std`` = mean and sample std (ddof=1)
      over seeds of each seed's cross-patient mean RMSE (run-to-run variability).
    * ``ablation_insulin_carbs.csv``: paired glucose-only vs. insulin/carbs comparison.

    The published-benchmark column is left empty for the project owner to fill in."""
    out = _results_dir(cfg)
    runs = _load_runs(cfg)
    seed_avg = runs.groupby(["model", "horizon_min", "patient"], sort=False).agg(
        n_windows=("n_windows", "first"), rmse=("rmse", "mean"), mae=("mae", "mean"),
        n_seeds=("seed", "nunique"), cohort=("cohort", "first")).reset_index()
    seed_avg.to_csv(out / "comparison_per_patient.csv", index=False)

    rows = []
    for cohort in _cohorts(cfg):
        sub, sub_raw = _subset(seed_avg, cohort), _subset(runs, cohort)
        for (model, horizon), g in sub.groupby(["model", "horizon_min"], sort=False):
            row = {"cohort": cohort, "model": model, "horizon_min": horizon, "n_patients": len(g),
                   "n_seeds": int(g["n_seeds"].iloc[0]), **summarize_patients(g)}
            if row["n_seeds"] > 1:
                per_seed = sub_raw[(sub_raw["model"] == model) & (sub_raw["horizon_min"] == horizon)]                     .groupby("seed").agg(rmse=("rmse", "mean"), mae=("mae", "mean"))
                row.update(rmse_seed_mean=per_seed["rmse"].mean(), rmse_seed_std=per_seed["rmse"].std(ddof=1),
                           mae_seed_mean=per_seed["mae"].mean(), mae_seed_std=per_seed["mae"].std(ddof=1))
            row["published_bglp_rmse"] = ""
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary["_o"] = summary["model"].map(_model_order)
    cohort_rank = {c: i for i, c in enumerate(_cohorts(cfg))}
    summary = summary.sort_values(["horizon_min", "cohort", "_o"], key=lambda c: c.map(cohort_rank) if c.name == "cohort" else c,
                                  kind="stable").drop(columns="_o").reset_index(drop=True)
    summary.to_csv(out / "comparison_summary.csv", index=False)
    write_ablation(cfg, runs)
    write_ablation_per_patient(cfg, seed_avg)
    return summary


def write_ablation(cfg: dict, runs: pd.DataFrame) -> pd.DataFrame:
    """Paired glucose-only vs. insulin/carbs comparison, per horizon and cohort.

    ``n_improved`` counts patients whose seed-averaged RMSE is lower with insulin/carbs (out of
    ``n_patients``); ``mean_delta`` is the mean over patients of (insulin/carbs - glucose-only),
    so negative means insulin/carbs helps. The per-seed columns show whether the conclusion
    holds seed by seed: ``delta_seed_mean`` / ``delta_seed_std`` are over the per-seed
    cross-patient mean differences, ``improved_per_seed`` lists n_improved for each seed."""
    glu, ins = runs[runs["model"] == GLUCOSE_LABEL], runs[runs["model"] == INSULIN_LABEL]
    rows = []
    if not glu.empty and not ins.empty:
        for horizon in sorted(glu["horizon_min"].unique()):
            for cohort in _cohorts(cfg):
                g = _subset(glu[glu["horizon_min"] == horizon], cohort)
                i = _subset(ins[ins["horizon_min"] == horizon], cohort)
                seeds = sorted(set(g["seed"]) & set(i["seed"]))
                if not seeds:
                    continue
                g, i = g[g["seed"].isin(seeds)], i[i["seed"].isin(seeds)]
                avg = i.groupby("patient")["rmse"].mean() - g.groupby("patient")["rmse"].mean()
                per_seed = [i[i["seed"] == sd].set_index("patient")["rmse"] - g[g["seed"] == sd].set_index("patient")["rmse"]
                            for sd in seeds]
                means = np.array([d.mean() for d in per_seed])
                rows.append({
                    "horizon_min": horizon, "cohort": cohort, "n_patients": len(avg), "n_seeds": len(seeds),
                    "n_improved": int((avg < 0).sum()), "mean_delta_rmse": avg.mean(),
                    "delta_seed_mean": means.mean(), "delta_seed_std": means.std(ddof=1) if len(means) > 1 else np.nan,
                    "improved_per_seed": ",".join(str(int((d < 0).sum())) for d in per_seed),
                })
    table = pd.DataFrame(rows)
    table.to_csv(_results_dir(cfg) / "ablation_insulin_carbs.csv", index=False)
    return table


def write_ablation_per_patient(cfg: dict, seed_avg: pd.DataFrame) -> pd.DataFrame | None:
    """Per patient: seed-averaged RMSE of both LSTM variants and their difference, next to how
    densely each patient logged meals and boluses (events per day, from the data-quality report,
    for the training and the test file). Purely descriptive: no test is applied to the columns."""
    out = _results_dir(cfg)
    quality_path = out / "data_quality.csv"
    if not quality_path.exists():
        return None
    q = pd.read_csv(quality_path)
    if "meals_per_day" not in q:  # report predates the event-count columns
        return None
    logging = q.pivot(index="patient", columns="split", values=["meals_per_day", "boluses_per_day"])
    logging.columns = [f"{metric}_{split}" for metric, split in logging.columns]
    rmse = seed_avg.pivot_table(index=["horizon_min", "patient", "cohort"], columns="model", values="rmse")
    if GLUCOSE_LABEL not in rmse or INSULIN_LABEL not in rmse:
        return None
    table = rmse[[GLUCOSE_LABEL, INSULIN_LABEL]].rename(
        columns={GLUCOSE_LABEL: "glucose_only_rmse", INSULIN_LABEL: "insulin_carbs_rmse"}).reset_index()
    table["delta_rmse"] = table["insulin_carbs_rmse"] - table["glucose_only_rmse"]
    table["improved"] = table["delta_rmse"] < 0
    table = table.merge(logging.reset_index(), on="patient").sort_values(["horizon_min", "cohort", "patient"])
    table.round(3).to_csv(out / "ablation_per_patient.csv", index=False)
    return table


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


def load_model(cfg: dict, checkpoint: str | Path, n_features: int):
    """Rebuild a trained model and its saved scaler from a checkpoint. Raises if the config's
    window / features / model / gap / split sections differ from the ones it was trained with."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("window", "features", "model", "gap", "split"):
        if cfg[key] != ckpt["cfg"][key]:
            raise ValueError(f"config section '{key}' differs from the one the checkpoint was trained with")
    model = build_model(cfg, n_features)
    model.load_state_dict(ckpt["model_state"])
    return model, Scaler(**ckpt["scaler"]), ckpt["seed"]  # scaler is never refit at evaluation time


def evaluate_checkpoint(cfg: dict, checkpoint: str | Path, name: str) -> pd.DataFrame:
    """Score a checkpoint on the test files with the scaler saved alongside it."""
    from src.data.dataset import feature_columns

    _, scaler, _ = load_model(cfg, checkpoint, len(feature_columns(cfg)))
    _, _, test, _ = build_datasets(cfg, scaler=scaler)
    x, y, patients, target_ts = collect_arrays(test)
    horizon_min = cfg["window"]["horizon"] * cfg["grid"]["step_min"]
    verify_fingerprint(cfg, horizon_min, patients, target_ts)

    model, _, seed = load_model(cfg, checkpoint, x.shape[2])
    pred = scaler.unscale_glucose(predict(model, x).numpy()).ravel()
    table = per_patient_metrics(pred, scaler.unscale_glucose(y.numpy()).ravel(), patients)
    table.insert(0, "model", model_label(cfg))
    table.insert(1, "horizon_min", horizon_min)
    table["seed"] = seed
    table.to_csv(_results_dir(cfg) / f"model_{name}_per_patient.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test files and refresh the tables.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="omit to only rebuild the comparison tables from existing results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.checkpoint:
        evaluate_checkpoint(cfg, args.checkpoint, Path(args.checkpoint).stem)  # run name = checkpoint name
    summary = write_comparison(cfg)
    h = cfg["window"]["horizon"] * cfg["grid"]["step_min"]
    shown = summary[summary["horizon_min"] == h].copy()
    shown["RMSE"] = shown.rmse_mean.round(2).astype(str) + " ± " + shown.rmse_std.round(2).astype(str)
    print(shown[["cohort", "model", "n_patients", "n_seeds", "RMSE"]].to_string(index=False))


if __name__ == "__main__":
    main()
