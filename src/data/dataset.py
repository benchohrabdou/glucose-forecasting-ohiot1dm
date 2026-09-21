"""Windowing, rejection rules, scaling and the torch Dataset.

A sample at grid index t has inputs = rows [t - window_len + 1 .. t] and target = glucose at
row t + horizon (mg/dL, unscaled by ``Scaler.unscale_glucose``). Time is in 5-minute bins.

Leakage rules enforced here:
  * chronological train/val cut; windows are built INSIDE each segment, so no window has
    inputs in one segment and a target in the other. The windows that would have straddled the
    cut are dropped and their number is reported as ``boundary_dropped`` in window_counts.csv;
  * glucose is re-filled per segment (causal forward-fill from real readings only);
  * the scaler is fit on training-segment rows only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocess import apply_gap_policy
from src.utils import get_logger, load_config

log = get_logger(__name__)

# Columns standardised with training statistics. glucose_observed (0/1) and time-of-day
# (already in [-1, 1]) are used as they are.
SCALED_COLUMNS = ("glucose", "bolus", "basal", "carbs")


def feature_columns(cfg: dict) -> list[str]:
    """Ordered model input columns. glucose_observed is always present so the model can tell
    forward-filled values from measured ones."""
    cols = ["glucose", "glucose_observed"]
    if cfg["features"]["insulin_carbs"]:
        cols += ["bolus", "basal", "carbs"]
    if cfg["features"]["time_of_day"]:
        cols += ["tod_sin", "tod_cos"]
    return cols


Segment = tuple[int, pd.DataFrame]  # (patient id, grid already filled for this segment)


def load_grids(processed_dir: str | Path, patients: Iterable[int], split: str) -> dict[int, pd.DataFrame]:
    """Read cached grids written by ``python -m src.data.preprocess``."""
    return {p: pd.read_pickle(Path(processed_dir) / f"{p}_{split}.pkl") for p in patients}


def chronological_split(grid: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First (1 - val_frac) of the time span -> train, the rest -> validation. Never shuffled:
    overlapping windows would otherwise leak near-identical samples across the boundary."""
    cut = int(round(len(grid) * (1 - val_frac)))
    return grid.iloc[:cut], grid.iloc[cut:]


def make_segments(cfg: dict, grids: dict[int, pd.DataFrame], part: str) -> list[Segment]:
    """Cut each patient's grid into the requested part ('train', 'val' or 'test') and refill
    glucose inside that segment only. 'test' grids are used whole (they are already a separate,
    later file)."""
    gap, step = cfg["gap"]["max_interp_gap_min"], cfg["grid"]["step_min"]
    segments = []
    for pid, grid in grids.items():
        if part in ("train", "val"):
            train, val = chronological_split(grid, cfg["split"]["val_frac"])
            grid = train if part == "train" else val
        elif part != "test":
            raise ValueError(f"unknown part: {part!r}")
        segments.append((pid, apply_gap_policy(grid, gap, step)))
    return segments


@dataclass
class Scaler:
    """Per-column mean/std, fit on training data only. Glucose is scaled explicitly so every
    reported metric can be converted back to mg/dL."""

    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(cls, segments: Iterable[Segment]) -> "Scaler":
        """Statistics over the given (training) segments. Glucose uses real readings only, so
        forward-filled repeats do not distort it."""
        frames = [g for _, g in segments]
        cat = pd.concat(frames)
        mean, std = {}, {}
        for col in SCALED_COLUMNS:
            values = cat.loc[cat["glucose_observed"], col] if col == "glucose" else cat[col]
            mean[col] = float(values.mean())
            std[col] = float(values.std(ddof=0)) or 1.0  # guard constant columns
        return cls(mean, std)

    def transform(self, grid: pd.DataFrame, columns: list[str]) -> np.ndarray:
        """Grid -> float32 array (n_rows, len(columns)); only SCALED_COLUMNS are standardised."""
        out = np.empty((len(grid), len(columns)), dtype=np.float32)
        for j, col in enumerate(columns):
            v = grid[col].to_numpy(dtype=np.float64)
            out[:, j] = (v - self.mean[col]) / self.std[col] if col in self.mean else v
        return out

    def scale_glucose(self, mgdl):
        return (mgdl - self.mean["glucose"]) / self.std["glucose"]

    def unscale_glucose(self, z):
        """Scaled -> mg/dL. Use this before computing any metric."""
        return z * self.std["glucose"] + self.mean["glucose"]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["std"])


def valid_window_ends(
    glucose: np.ndarray, observed: np.ndarray, window_len: int, horizon: int, max_imputed_frac: float
) -> tuple[np.ndarray, dict]:
    """Indices t whose window passes every rejection rule, plus rejection counts.

    Rules, applied in this order (each rejected window is counted once):
      1. target bin t + horizon is not a real CGM reading (never score an imputed target);
      2. any input glucose in [t - window_len + 1, t] is still NaN after filling;
      3. the fraction of forward-filled (imputed) input bins exceeds max_imputed_frac.
    """
    n = len(glucose)
    t = np.arange(window_len - 1, n - horizon)
    nan_cum = np.concatenate([[0], np.cumsum(np.isnan(glucose))])
    imp_cum = np.concatenate([[0], np.cumsum(~observed & ~np.isnan(glucose))])
    lo = t - window_len + 1
    n_nan = nan_cum[t + 1] - nan_cum[lo]
    frac_imputed = (imp_cum[t + 1] - imp_cum[lo]) / window_len

    bad_target = ~observed[t + horizon]
    bad_nan = ~bad_target & (n_nan > 0)
    bad_imputed = ~bad_target & ~bad_nan & (frac_imputed > max_imputed_frac)
    keep = ~(bad_target | bad_nan | bad_imputed)
    counts = {
        "candidates": len(t),
        "rejected_target_not_observed": int(bad_target.sum()),
        "rejected_nan_input": int(bad_nan.sum()),
        "rejected_imputed_frac": int(bad_imputed.sum()),
        "kept": int(keep.sum()),
    }
    return t[keep], counts


class GlucoseWindowDataset(Dataset):
    """Yields ``(x, y, meta)``: x FloatTensor[window_len, n_features] (scaled), y FloatTensor[1]
    (scaled glucose at t + horizon), meta = {patient, target_ts, input_end_ts} with timestamps
    as int64 ns since epoch so the default DataLoader collate works."""

    def __init__(
        self,
        segments: list[Segment],
        scaler: Scaler,
        columns: list[str],
        window_len: int,
        horizon: int,
        max_imputed_frac: float,
        step_min: int = 5,
    ) -> None:
        self.window_len, self.horizon, self.step_min = window_len, horizon, step_min
        self._feats, self._y, self._ts, self._patient = [], [], [], []
        self._samples: list[tuple[int, int]] = []  # (segment idx, t)
        self.counts: list[dict] = []
        for si, (pid, grid) in enumerate(segments):
            glucose = grid["glucose"].to_numpy(dtype=np.float64)
            observed = grid["glucose_observed"].to_numpy(dtype=bool)
            ends, counts = valid_window_ends(glucose, observed, window_len, horizon, max_imputed_frac)
            self.counts.append({"patient": pid, **counts})
            self._feats.append(scaler.transform(grid, columns))
            self._y.append(scaler.scale_glucose(glucose).astype(np.float32))
            self._ts.append(grid.index.to_numpy())
            self._patient.append(pid)
            self._samples += [(si, int(t)) for t in ends]

    def __len__(self) -> int:
        return len(self._samples)

    def timestamps(self, i: int) -> pd.DatetimeIndex:
        """Timestamps of the input rows of sample i (for leakage checks)."""
        si, t = self._samples[i]
        return pd.DatetimeIndex(self._ts[si][t - self.window_len + 1 : t + 1])

    def __getitem__(self, i: int):
        si, t = self._samples[i]
        x = torch.from_numpy(self._feats[si][t - self.window_len + 1 : t + 1].copy())
        y = torch.tensor([self._y[si][t + self.horizon]])
        meta = {
            "patient": self._patient[si],
            "target_ts": int(self._ts[si][t + self.horizon].astype("datetime64[ns]").astype(np.int64)),
            "input_end_ts": int(self._ts[si][t].astype("datetime64[ns]").astype(np.int64)),
        }
        return x, y, meta


def build_datasets(cfg: dict, patients: list[int] | None = None):
    """(train, val, test datasets, scaler). Scaler is fit on training segments only; window
    counts are persisted to ``<results_dir>/window_counts.csv``."""
    all_patients = [p for ps in cfg["cohorts"].values() for p in ps]
    patients = [p for p in all_patients if not patients or p in patients]
    proc = cfg["paths"]["processed_dir"]
    w = cfg["window"]
    columns = feature_columns(cfg)

    train_grids = load_grids(proc, patients, "train")
    train_seg = make_segments(cfg, train_grids, "train")
    scaler = Scaler.fit(train_seg)  # training rows only: no val/test statistics
    parts = {
        "train": train_seg,
        "val": make_segments(cfg, train_grids, "val"),
        "test": make_segments(cfg, load_grids(proc, patients, "test"), "test"),
    }
    datasets, rows = {}, []
    for name, seg in parts.items():
        ds = GlucoseWindowDataset(
            seg, scaler, columns, w["window_len"], w["horizon"], w["max_imputed_frac"], cfg["grid"]["step_min"]
        )
        datasets[name] = ds
        rows += [{"split": name, **c} for c in ds.counts]
    # Windows that would span the train/val cut = candidates of the uncut grid minus the two
    # segments' candidates. They exist in neither dataset.
    span = w["window_len"] + w["horizon"] - 1
    cand = {(r["split"], r["patient"]): r["candidates"] for r in rows}
    for r in rows:
        if r["split"] == "val":
            full = len(train_grids[r["patient"]]) - span
            r["boundary_dropped"] = full - cand[("train", r["patient"])] - r["candidates"]
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "window_counts.csv", index=False)
    return datasets["train"], datasets["val"], datasets["test"], scaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets and write window counts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, val, test, _ = build_datasets(cfg)
    log.info("windows: train=%d val=%d test=%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
