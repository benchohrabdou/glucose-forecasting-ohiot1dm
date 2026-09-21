"""XML -> regular 5-minute grid, with the glucose gap policy.

Units and time conventions: glucose in mg/dL, insulin in U per bin, carbs in grams
per bin, one row per ``step_min`` (default 5) minutes, naive de-identified local time.
Every stream is snapped to the *nearest* grid point so that a bolus, a meal and a CGM
reading that happened in the same instant land in the same bin.

Columns: glucose, glucose_observed, bolus, basal, carbs [, tod_sin, tod_cos].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.parse import load_patient
from src.utils import get_logger, load_config

log = get_logger(__name__)

# split name -> file suffix
SPLIT_SUFFIX = {"train": "training", "test": "testing"}


def raw_path(data_root: str | Path, year: str, patient: int, split: str) -> Path:
    """Path of ``<data_root>/<year>/<split>/<id>-ws-<training|testing>.xml``."""
    return Path(data_root) / str(year) / split / f"{patient}-ws-{SPLIT_SUFFIX[split]}.xml"


def _snap(ts: pd.Series, step_min: int) -> pd.Series:
    """Snap timestamps to the nearest grid point (jitter is seconds-level)."""
    return ts.dt.round(f"{step_min}min")


def _in_grid(ts: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    return (ts >= index[0]) & (ts <= index[-1])


def _cgm_on_grid(glucose: pd.DataFrame, step_min: int) -> tuple[pd.Series, int]:
    """CGM readings on a complete grid (NaN where absent) and the number of bin collisions."""
    g = glucose.sort_values("ts", kind="stable")
    snapped = _snap(g["ts"], step_min)
    keep = ~snapped.duplicated(keep="first").to_numpy()  # keep the earlier reading on a collision
    collisions = int((~keep).sum())
    series = pd.Series(
        g["value"].astype(float).to_numpy()[keep],
        index=pd.DatetimeIndex(snapped.to_numpy()[keep]),
    )
    index = pd.date_range(series.index.min(), series.index.max(), freq=f"{step_min}min")
    return series.reindex(index), collisions


def _bolus_per_bin(bolus: pd.DataFrame, index: pd.DatetimeIndex, step_min: int) -> tuple[pd.Series, int]:
    """Insulin (U) per bin. Instant boluses land in one bin; extended ones (ts_end > ts_begin,
    i.e. square / square dual) are spread uniformly over every bin from begin to end inclusive."""
    out = pd.Series(0.0, index=index)
    if bolus.empty:
        return out, 0
    begin, end = _snap(bolus["ts_begin"], step_min), _snap(bolus["ts_end"], step_min)
    outside = int((~_in_grid(begin, index)).sum())
    step = pd.Timedelta(minutes=step_min)
    for b, e, dose in zip(begin, end, bolus["dose"].fillna(0.0)):
        n_bins = int((e - b) / step) + 1 if e > b else 1
        bins = pd.date_range(b, periods=n_bins, freq=f"{step_min}min")
        # Clip to the grid without renormalising: doses outside the CGM span are unusable.
        bins = bins[(bins >= index[0]) & (bins <= index[-1])]
        out.loc[bins] += dose / n_bins
    return out, outside


def _carbs_per_bin(meal: pd.DataFrame, index: pd.DatetimeIndex, step_min: int) -> tuple[pd.Series, int]:
    """Self-reported carbs (g) per bin. All meal types are kept, including HypoCorrection:
    they are real carb intake."""
    out = pd.Series(0.0, index=index)
    if meal.empty:
        return out, 0
    ts = _snap(meal["ts"], step_min)
    inside = _in_grid(ts, index)
    per_bin = meal.loc[inside, "carbs"].fillna(0.0).groupby(ts[inside]).sum()
    out.loc[per_bin.index] += per_bin
    return out, int((~inside).sum())


def _basal_per_bin(
    basal: pd.DataFrame,
    temp_basal: pd.DataFrame,
    index: pd.DatetimeIndex,
    step_min: int,
    initial_basal: float | None,
) -> tuple[pd.Series, int, float]:
    """Basal insulin (U per bin) = rate (U/hr) * step/60, with temp basals overriding the
    standing rate over [ts_begin, ts_end); value 0 means the pump is suspended.

    Returns (per-bin insulin, number of bins whose standing rate was unknown, last standing
    rate in the file). Bins before the file's first basal event have no known rate; they get
    ``initial_basal`` (the previous file's last rate, which is causal because training strictly
    precedes testing) or 0.0 when not given."""
    ts = _snap(basal["ts"], step_min)
    rate = pd.Series(basal["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(ts.to_numpy()))
    rate = rate[~rate.index.duplicated(keep="last")].sort_index()  # last event in a bin wins
    # Last event by TIME (not file order): this is the rate the next file starts under.
    final_rate = float(rate.iloc[-1])
    # ffill on the union so events *before* the first CGM reading still set the standing rate.
    rate = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    unknown = int(rate.isna().sum())
    rate = rate.fillna(0.0 if initial_basal is None else initial_basal)

    if not temp_basal.empty:
        tb = temp_basal.assign(
            b=_snap(temp_basal["ts_begin"], step_min), e=_snap(temp_basal["ts_end"], step_min)
        ).sort_values("b", kind="stable")
        for b, e, value in zip(tb["b"], tb["e"], tb["value"]):
            if pd.notna(value):
                rate[(rate.index >= b) & (rate.index < e)] = value
    return rate * step_min / 60.0, unknown, final_rate


def build_grid(
    tables: dict[str, pd.DataFrame],
    step_min: int = 5,
    initial_basal: float | None = None,
    time_of_day: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Raw tables -> grid with glucose still un-imputed (NaN where no reading), plus counts.

    Spans first to last CGM reading of the file. ``glucose_observed`` marks bins with a real
    reading; it is fixed here, before any imputation, so targets are never scored when imputed.
    Insulin and carbs are event streams: absence means 0, never interpolated."""
    glucose, collisions = _cgm_on_grid(tables["glucose_level"], step_min)
    index = glucose.index
    bolus, bolus_out = _bolus_per_bin(tables["bolus"], index, step_min)
    carbs, meal_out = _carbs_per_bin(tables["meal"], index, step_min)
    basal, basal_unknown, final_rate = _basal_per_bin(
        tables["basal"], tables["temp_basal"], index, step_min, initial_basal
    )
    grid = pd.DataFrame(
        {
            "glucose": glucose,
            "glucose_observed": glucose.notna(),
            "bolus": bolus,
            "basal": basal,
            "carbs": carbs,
        }
    )
    if time_of_day:
        minutes = index.hour * 60 + index.minute
        grid["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
        grid["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    stats = {
        "n_bins": len(grid),
        "n_cgm_readings": int(grid["glucose_observed"].sum()),
        "cgm_collisions": collisions,
        "bolus_outside_grid": bolus_out,
        "meal_outside_grid": meal_out,
        # Logged events that fall inside the grid (all meal types incl. HypoCorrection; every bolus
        # type). Counts, not doses: used for the meals/day and boluses/day data-quality columns.
        "meals_in_grid": len(tables["meal"]) - meal_out,
        "boluses_in_grid": len(tables["bolus"]) - bolus_out,
        "basal_unknown_bins": basal_unknown,
        "final_basal_rate": final_rate,
    }
    return grid, stats


def ffill_causal(glucose: pd.Series, max_gap_steps: int) -> pd.Series:
    """Forward-fill at most ``max_gap_steps`` bins after a reading. The value at bin t depends
    only on readings at or before t. Linear interpolation is deliberately NOT used anywhere,
    not even for training: it makes a bin near a window's end encode a reading that arrives
    after the window, and trains the model on a fill pattern it never sees at test time."""
    return glucose.ffill(limit=max_gap_steps)


def apply_gap_policy(grid: pd.DataFrame, max_interp_gap_min: int, step_min: int = 5) -> pd.DataFrame:
    """Fill glucose with the one causal policy used for train, validation and test alike.

    Always starts from ``glucose.where(glucose_observed)``, so it is idempotent and can be
    re-applied to any chronological segment (dataset.py does this after cutting train/val, so
    no reading from the other side of the cut fills a bin). Runs longer than the limit are
    filled for their first ``limit`` bins, then stay NaN."""
    out = grid.copy()
    raw = grid["glucose"].where(grid["glucose_observed"])
    out["glucose"] = ffill_causal(raw, max_interp_gap_min // step_min)
    return out


def preprocess_patient(
    cfg: dict, year: str, patient: int, split: str, initial_basal: float | None = None
) -> tuple[pd.DataFrame, dict]:
    """One patient file -> filled grid + stats."""
    _, tables = load_patient(raw_path(cfg["paths"]["data_root"], year, patient, split))
    step = cfg["grid"]["step_min"]
    if initial_basal == 0:
        # A zero seed would mean carrying a pump suspension into the next file.
        log.warning("%s %s: seeded basal rate is 0 (suspension?)", patient, split)
    grid, stats = build_grid(tables, step, initial_basal, cfg["features"]["time_of_day"])
    if stats["cgm_collisions"]:
        log.warning("%s %s: %d CGM bin collisions (kept first)", patient, split, stats["cgm_collisions"])
    return apply_gap_policy(grid, cfg["gap"]["max_interp_gap_min"], step), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache 5-minute grids for OhioT1DM patients.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--patients", type=int, nargs="*", help="restrict to these patient ids")
    args = parser.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year, patients in cfg["cohorts"].items():
        for pid in patients:
            if args.patients and pid not in args.patients:
                continue
            train, tr_stats = preprocess_patient(cfg, year, pid, "train")
            # Test basal at the file start is unknown; carry over the last training rate (causal).
            test, te_stats = preprocess_patient(cfg, year, pid, "test", tr_stats["final_basal_rate"])
            for split, grid, stats in (("train", train, tr_stats), ("test", test, te_stats)):
                grid.to_pickle(out_dir / f"{pid}_{split}.pkl")
                rows.append({"year": year, "patient": pid, "split": split, **stats})
                log.info("%s %s: %d bins, %d CGM readings", pid, split, stats["n_bins"], stats["n_cgm_readings"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(results_dir / "preprocess_stats.csv", index=False)


if __name__ == "__main__":
    main()
