"""Aggregate data-quality report over the cached grids.

Writes ``results/data_quality.csv``: one row per patient and split, aggregate statistics
only (no raw records). Run after ``python -m src.data.preprocess``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config

log = get_logger(__name__)

HYPO, HYPER = 70, 180  # mg/dL; in-range is [70, 180]


def gap_lengths(observed: pd.Series) -> pd.Series:
    """Length (in bins) of every run of missing CGM readings."""
    missing = ~observed
    run_id = (missing != missing.shift()).cumsum()
    return missing.groupby(run_id).sum()[missing.groupby(run_id).first()]


def summarize(grid: pd.DataFrame, step_min: int, max_interp_gap_min: int) -> dict:
    """Aggregate quality metrics for one grid. Glycemic ranges use real readings only."""
    observed = grid["glucose_observed"]
    real = grid.loc[observed, "glucose"]
    gaps = gap_lengths(observed)
    days = len(grid) * step_min / 1440
    return {
        "days": round(days, 2),
        "cgm_coverage_pct": round(100 * observed.mean(), 2),
        "n_gaps": len(gaps),
        "longest_gap_min": int(gaps.max() * step_min) if len(gaps) else 0,
        "gaps_fillable": int((gaps * step_min <= max_interp_gap_min).sum()),
        "gaps_too_long": int((gaps * step_min > max_interp_gap_min).sum()),
        "imputed_bins_pct": round(100 * (grid["glucose"].notna() & ~observed).mean(), 2),
        "glucose_mean": round(real.mean(), 1),
        "glucose_std": round(real.std(), 1),
        "glucose_min": real.min(),
        "glucose_max": real.max(),
        "pct_hypo": round(100 * (real < HYPO).mean(), 2),
        "pct_in_range": round(100 * real.between(HYPO, HYPER).mean(), 2),
        "pct_hyper": round(100 * (real > HYPER).mean(), 2),
        "bolus_u_per_day": round(grid["bolus"].sum() / days, 1),
        "basal_u_per_day": round(grid["basal"].sum() / days, 1),
        "carbs_g_per_day": round(grid["carbs"].sum() / days, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate data-quality report.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results_dir"])
    stats = pd.read_csv(results_dir / "preprocess_stats.csv")

    rows = []
    for _, s in stats.iterrows():
        grid = pd.read_pickle(Path(cfg["paths"]["processed_dir"]) / f"{s['patient']}_{s['split']}.pkl")
        summary = summarize(grid, cfg["grid"]["step_min"], cfg["gap"]["max_interp_gap_min"])
        rows.append(
            {
                "year": s["year"],
                "patient": s["patient"],
                "split": s["split"],
                **summary,
                "meals_per_day": round(s["meals_in_grid"] / summary["days"], 2),
                "boluses_per_day": round(s["boluses_in_grid"] / summary["days"], 2),
                "cgm_collisions": s["cgm_collisions"],
                "bolus_outside_grid": s["bolus_outside_grid"],
                "meal_outside_grid": s["meal_outside_grid"],
                "basal_unknown_bins": s["basal_unknown_bins"],
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(results_dir / "data_quality.csv", index=False)
    log.info("wrote %s (%d rows)", results_dir / "data_quality.csv", len(report))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
