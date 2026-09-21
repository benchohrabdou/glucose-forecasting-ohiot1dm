"""Figures for the step-6 analysis (matplotlib, PNG, light theme).

Colour follows the model in every figure, so a model never changes colour between charts:
LSTM with insulin/carbs = palette slot 1, LSTM glucose-only = slot 2, ridge = slot 3 (this
three-slot set was checked with the dataviz palette validator, all-pairs, light mode). Persistence
is the reference and is a neutral dashed grey, not a competing hue. Slot 3 (aqua) is below 3:1
contrast on the surface, so every chart carries a legend and every figure has a matching CSV table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis import HYPER, HYPO, RANGES, representative_window  # noqa: E402
from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STYLE = {
    "persistence": dict(color="#6b6a66", ls=(0, (4, 3)), label="Persistence"),
    "ridge": dict(color="#1baf7a", ls="-", label="Ridge"),
    GLUCOSE_LABEL: dict(color="#eb6834", ls="-", label="LSTM, glucose only"),
    INSULIN_LABEL: dict(color="#2a78d6", ls="-", label="LSTM, glucose + insulin/carbs"),
}
ORDER = ["persistence", "ridge", GLUCOSE_LABEL, INSULIN_LABEL]


def _axes(fig, ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _shared_legend(fig, ax, ncol: int) -> None:
    """One legend under the plot area: it never sits on top of data, and the models keep the same
    key in every panel."""
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.01))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))


def _save(fig, path: Path) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def predicted_vs_actual(sets: dict, path: Path) -> None:
    """One representative 8 h stretch of one test file per horizon (selection rule in
    ``representative_window``). Forecasts are plotted at the time they predict, so a forecast that
    only repeats the last value appears as a delayed copy of the actual trace."""
    fig, axes = plt.subplots(len(sets), 1, figsize=(9, 3.6 * len(sets)))
    for ax, (_, ps) in zip(np.atleast_1d(axes), sets.items()):
        patient, rows = representative_window(ps)
        t = (ps.meta["target_ts"].iloc[rows] - ps.meta["target_ts"].iloc[rows[0]]).dt.total_seconds() / 3600
        ax.axhspan(HYPO, HYPER, color=GRID, alpha=0.6, lw=0)  # in-range band, recessive
        ax.plot(t, ps.meta["y"].to_numpy()[rows], color=INK, lw=1.8, label="Actual (CGM)")
        for model in (m for m in ORDER if m in (GLUCOSE_LABEL, INSULIN_LABEL, "persistence")):
            s = STYLE[model]
            ax.plot(t, ps.preds[(model, 0)][rows], color=s["color"], ls=s["ls"], lw=1.4, label=s["label"])
        _axes(fig, ax, f"{ps.horizon_min}-minute forecast: patient {patient}, 8 h of the test file (LSTM seed 0)",
              "hours since start of stretch", "glucose (mg/dL)")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=4)
    _save(fig, path)


def residuals(sets: dict, path: Path) -> None:
    """Density of prediction error over all scored windows; LSTMs shown for seed 0."""
    edges = np.arange(-150, 151, 5)
    fig, axes = plt.subplots(1, len(sets), figsize=(5.6 * len(sets), 3.8), sharey=True)
    for ax, (_, ps) in zip(np.atleast_1d(axes), sets.items()):
        err = {m: ps.preds[(m, 0)] - ps.meta["y"].to_numpy() for m in ORDER}
        for m in ORDER:
            dens, _ = np.histogram(err[m], bins=edges, density=True)
            ax.stairs(dens, edges, color=STYLE[m]["color"], ls=STYLE[m]["ls"], lw=1.6, label=STYLE[m]["label"])
        ax.axvline(0, color=INK_2, lw=0.8)
        _axes(fig, ax, f"{ps.horizon_min}-minute forecast", "prediction error, predicted - actual (mg/dL); beyond ±150 not shown",
              "density")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=4)
    _save(fig, path)


def error_by_range(range_df: pd.DataFrame, path: Path) -> None:
    """RMSE by actual glycemic range, all patients pooled; x labels carry the reading counts."""
    horizons = sorted(range_df["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.8 * len(horizons), 4.0))
    for ax, h in zip(np.atleast_1d(axes), horizons):
        d = range_df[(range_df["horizon_min"] == h) & (range_df["cohort"] == "all")]
        width = 0.2
        for i, m in enumerate(ORDER):
            v = d[d["model"] == m].set_index("range").loc[list(RANGES), "rmse"].to_numpy()
            ax.bar(np.arange(3) + (i - 1.5) * width, v, width, color=STYLE[m]["color"], label=STYLE[m]["label"],
                   edgecolor=SURFACE, linewidth=1.5)  # surface-coloured edge = the 2px gap between bars
        n = d[d["model"] == "persistence"].set_index("range").loc[list(RANGES), "n_readings"]
        ax.set_xticks(np.arange(3), [f"{r}\nn = {n[r]:,}" for r in RANGES])
        _axes(fig, ax, f"{h}-minute forecast", "actual glucose range (scored readings)", "RMSE (mg/dL)")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=4)
    _save(fig, path)


def lag_curves(curves: pd.DataFrame, lag: pd.DataFrame, path: Path) -> None:
    """RMSE of the forecast for time T against the actual glucose at T - lag. The dot marks each
    model's best lag; the dotted line is the forecast horizon (the lag of a repeat-last-value forecast)."""
    horizons = sorted(curves["horizon_min"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.8 * len(horizons), 3.9))
    for ax, h in zip(np.atleast_1d(axes), horizons):
        for m in ORDER:
            c = curves[(curves["horizon_min"] == h) & (curves["model"] == m)]
            s = STYLE[m]
            ax.plot(c["lag_min"], c["rmse"], color=s["color"], ls=s["ls"], lw=1.6, label=s["label"])
            best = c.loc[c["rmse"].idxmin()]
            ax.plot(best["lag_min"], best["rmse"], "o", ms=8, color=s["color"], mec=SURFACE, mew=1.5)
        ax.axvline(h, color=INK_2, lw=0.9, ls=":")
        ax.text(h, ax.get_ylim()[1], " horizon", color=INK_2, fontsize=8.5, va="top")
        _axes(fig, ax, f"{h}-minute forecast", "lag between forecast and actual glucose (min)", "RMSE (mg/dL)")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=4)
    _save(fig, path)


def make_figures(cfg: dict, sets: dict, result: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    predicted_vs_actual(sets, fig_dir / "predicted_vs_actual.png")
    residuals(sets, fig_dir / "residual_distributions.png")
    error_by_range(result["range"], fig_dir / "error_by_range.png")
    lag_curves(result["curves"], result["lag"][result["lag"]["cohort"] == "all"], fig_dir / "lag_curves.png")
