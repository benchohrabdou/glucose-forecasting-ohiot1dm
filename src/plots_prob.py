"""Aggregate figures for the probabilistic models (Section 4.4). Same visual system as src/plots.py:
colour follows the input variant (glucose only = orange, glucose + insulin/carbs = blue), neutral
greys for metrics that are not model identities, a legend on every chart, and a CSV behind each one.
Pooled, aggregate numbers only: no patient traces."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL  # noqa: E402
from src.plots import INK, INK_2, SURFACE, _axes, _save, _shared_legend  # noqa: E402
from src.probabilistic import C_GLU, C_INS, Q_GLU, Q_INS, RANGES  # noqa: E402

VARIANT_COLOR = {Q_GLU: "#eb6834", Q_INS: "#2a78d6"}
SHORT = {"persistence": "Persistence", GLUCOSE_LABEL: "MSE LSTM\nglucose", INSULIN_LABEL: "MSE LSTM\n+ ins/carbs",
         Q_GLU: "Quantile LSTM\nglucose", Q_INS: "Quantile LSTM\n+ ins/carbs",
         C_GLU: "Classifier\nglucose", C_INS: "Classifier\n+ ins/carbs"}
ORDER = list(SHORT)


def coverage_by_range(cal: pd.DataFrame, path: Path) -> None:
    """Empirical coverage of the 80% and 90% intervals by actual range; dashed lines = nominal."""
    horizons = sorted(cal["horizon_min"].unique())
    ranges = list(RANGES)
    fig, axes = plt.subplots(1, len(horizons), figsize=(6.2 * len(horizons), 4.2), sharey=True)
    width = 0.2
    for ax, h in zip(np.atleast_1d(axes), horizons):
        d = cal[cal["horizon_min"] == h]
        i = 0
        for model in (Q_GLU, Q_INS):
            for iv, hatch in (("80%", "///"), ("90%", "")):
                v = d[(d["model"] == model) & (d["interval"] == iv)].set_index("range").loc[ranges, "coverage"]
                ax.bar(np.arange(len(ranges)) + (i - 1.5) * width, v, width, color=VARIANT_COLOR[model],
                       alpha=0.55 if iv == "80%" else 1.0, hatch=hatch, edgecolor=SURFACE, linewidth=1.5,
                       label=f"{SHORT[model].replace(chr(10), ' ')}, {iv} interval")
                i += 1
        for nominal in (0.8, 0.9):
            ax.axhline(nominal, color=INK_2, lw=0.9, ls=(0, (4, 3)))
        n = d[(d["model"] == Q_GLU) & (d["interval"] == "80%")].set_index("range").loc[ranges, "n_readings"]
        ax.set_xticks(np.arange(len(ranges)), [f"{r}\nn = {n[r]:,}" for r in ranges])
        ax.set_ylim(0, 1.0)
        _axes(fig, ax, f"{h}-minute forecast: interval coverage (dashed = nominal 80% / 90%)",
              "actual glucose range (scored test readings)", "empirical coverage")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=2)
    _save(fig, path)


def alerting_comparison(alert: pd.DataFrame, path: Path) -> None:
    """Test sensitivity and precision of each model with its validation-chosen alert rule; dot = F2."""
    horizons = sorted(alert["horizon_min"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(10, 3.6 * len(horizons)))
    for ax, h in zip(np.atleast_1d(axes), horizons):
        d = alert[alert["horizon_min"] == h].set_index("model")
        models = [m for m in ORDER if m in d.index]
        x = np.arange(len(models))
        ax.bar(x - 0.18, d.loc[models, "sensitivity"], 0.34, color="#52514e", edgecolor=SURFACE, linewidth=1.5,
               label="Sensitivity")
        ax.bar(x + 0.18, d.loc[models, "precision"].fillna(0), 0.34, color="#b9b8b2", edgecolor=SURFACE,
               linewidth=1.5, label="Precision")
        ax.plot(x, d.loc[models, "f2"], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, label="F2")
        ax.set_xticks(x, [SHORT[m] for m in models], fontsize=8.5)
        ax.set_ylim(0, 1.0)
        n = int(d["n_test_lows"].iloc[0])
        _axes(fig, ax, f"{h}-minute forecast: hypoglycemia alerts on test (rules chosen on validation; {n} actual lows)",
              "", "rate")
    fig.tight_layout()
    _shared_legend(fig, np.atleast_1d(axes)[0], ncol=3)
    _save(fig, path)


def make_figures(res: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not res["calibration"].empty:
        coverage_by_range(res["calibration"], fig_dir / "quantile_coverage_by_range.png")
    alerting_comparison(res["alerting"], fig_dir / "alerting_comparison.png")
