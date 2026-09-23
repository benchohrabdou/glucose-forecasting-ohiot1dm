"""Step-6 evaluation extras, all on the exact test windows the headline tables use.

Writes to results/ (aggregate tables) and results/figures/:
  error_by_range*.csv      RMSE / MAE / bias by glycemic range, with reading counts
  clarke_zones.csv         Clarke error-grid zone percentages
  lag_and_anticipation.csv does a model anticipate changes, or repeat the last value later?
  lag_curves.csv           RMSE as a function of the lag between forecast and glucose
  residual_summary.csv     bias / spread / tails of the errors
  calibration_reverse_slope.csv  actual change regressed on predicted change
  hypo_detection.csv       sensitivity / precision of "forecast < 70 / 80 / 90" for "actual < 70"
  forecast_spread.csv      spread of forecasts vs actual glucose; share of forecasts below 70

``python -m src.analysis --config configs/base.yaml [--rebuild]``

All errors are in mg/dL, computed on windows whose target is a real CGM reading. Models with
several seeds (the LSTMs) are scored per seed and the metric is then averaged over seeds.
Predictions are cached in <processed_dir>/predictions_ph<min>.pkl (patient-level, gitignored).
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import build_datasets, collect_arrays, feature_columns, load_grids, make_segments
from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL, _results_dir, cohort_of, load_model, predict, verify_fingerprint
from src.models.baselines import RidgeRegression
from src.utils import get_logger, load_config

log = get_logger(__name__)

HYPO, HYPER = 70.0, 180.0          # mg/dL: hypo < 70, in range 70-180 inclusive, hyper > 180
RANGES = ("hypo (<70)", "in range (70-180)", "hyper (>180)")
LOW_N = 30                          # ranges with fewer scored readings than this are flagged
LARGE_MOVE = 10.0                   # mg/dL: a "real" glucose change for the direction check
LAG_EXTRA_STEPS = 6                 # lags searched: 0 .. horizon + this many bins
MODELS = ("persistence", "ridge", GLUCOSE_LABEL, INSULIN_LABEL)


@dataclass
class PredictionSet:
    """Test-window predictions of every model at one horizon (all in mg/dL).

    ``meta`` has one row per scored window: patient, target_ts, y (actual target), last (last
    input glucose), cohort. ``preds[(model, seed)]`` aligns with those rows; deterministic models
    use seed 0."""

    horizon_min: int
    meta: pd.DataFrame
    preds: dict[tuple[str, int], np.ndarray]

    def seeds(self, model: str) -> list[int]:
        return sorted(s for (m, s) in self.preds if m == model)


# ------------------------------------------------------------------ building predictions

def build_prediction_set(cfg: dict, horizon: int, config_dir: str | Path = "configs") -> PredictionSet:
    """Persistence, ridge and every trained LSTM seed, scored on identical windows.

    Ridge is refit at the alpha the baselines chose on VALIDATION. Each variant's windows are
    checked against the baselines' fingerprint, so a mismatch fails loudly."""
    step = cfg["grid"]["step_min"]
    minutes = horizon * step
    variants = {
        GLUCOSE_LABEL: load_config(Path(config_dir) / f"lstm_ph{minutes}.yaml"),
        INSULIN_LABEL: load_config(Path(config_dir) / f"lstm_ins_ph{minutes}.yaml"),
    }
    base = variants[GLUCOSE_LABEL]
    train, _, test, scaler = build_datasets(base)
    xtr, ytr, _, _ = collect_arrays(train)
    x, y, patients, target_ts = collect_arrays(test)
    verify_fingerprint(base, minutes, patients, target_ts)
    unscale = lambda z: scaler.unscale_glucose(np.asarray(z)).ravel()  # noqa: E731

    cohorts = cohort_of(cfg)
    meta = pd.DataFrame({
        "patient": patients, "target_ts": pd.to_datetime(target_ts), "y": unscale(y.numpy()),
        "last": unscale(x[:, -1, 0].numpy()), "cohort": [cohorts[p] for p in patients],
    })
    preds: dict[tuple[str, int], np.ndarray] = {("persistence", 0): meta["last"].to_numpy().copy()}

    val = pd.read_csv(_results_dir(cfg) / "baselines_validation.csv")
    val = val[(val["model"] == "ridge") & (val["horizon_min"] == minutes)]
    alpha = float(val.loc[val["val_rmse"].idxmin(), "param"].split("=")[1])
    preds[("ridge", 0)] = unscale(RidgeRegression(alpha).fit(xtr, ytr)(x).numpy())

    for label, vcfg in variants.items():
        stem = f"lstm{'_ins' if vcfg['features']['insulin_carbs'] else ''}_ph{minutes}"
        ckpts = sorted(Path(vcfg["paths"]["checkpoint_dir"]).glob(f"{stem}_seed*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"no checkpoints for {stem}: run `python -m src.run_seeds` first")
        for ckpt in ckpts:
            model, ck_scaler, seed = load_model(vcfg, ckpt, len(feature_columns(vcfg)))
            _, _, vtest, _ = build_datasets(vcfg, scaler=ck_scaler)
            vx, _, vp, vt = collect_arrays(vtest)
            if not (np.array_equal(vp, patients) and np.array_equal(vt, target_ts)):
                raise AssertionError(f"{ckpt.name}: test windows differ from the baselines'")
            preds[(label, seed)] = ck_scaler.unscale_glucose(predict(model, vx).numpy()).ravel()
    return PredictionSet(minutes, meta, preds)


def load_or_build(cfg: dict, horizon: int, rebuild: bool = False) -> PredictionSet:
    """Cached prediction set. The cache holds plain data (a dict), not a pickled class, so it can
    be read no matter which module wrote it (a class pickled from ``python -m`` is __main__.X)."""
    path = Path(cfg["paths"]["processed_dir"]) / f"predictions_ph{horizon * cfg['grid']['step_min']}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return PredictionSet(d["horizon_min"], d["meta"], d["preds"])
    ps = build_prediction_set(cfg, horizon)
    with open(path, "wb") as f:
        pickle.dump({"horizon_min": ps.horizon_min, "meta": ps.meta, "preds": ps.preds}, f)
    return ps


# ------------------------------------------------------------------ glycemic range

def glycemic_range(y: np.ndarray) -> np.ndarray:
    """0 = hypo (< 70), 1 = in range (70-180 inclusive), 2 = hyper (> 180), by the ACTUAL value."""
    return np.where(y < HYPO, 0, np.where(y > HYPER, 2, 1))


def _cohort_masks(ps: PredictionSet, cfg: dict) -> dict[str, np.ndarray]:
    c = ps.meta["cohort"].to_numpy()
    return {"all": np.ones(len(c), bool), **{k: c == k for k in sorted(set(cohort_of(cfg).values()))}}


def _err_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    e = pred - y
    return {"rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e))}


def _seed_avg(ps: PredictionSet, model: str, fn) -> dict:
    """fn(pred) -> dict of metrics for one seed; returns their mean over seeds (+ rmse spread)."""
    rows = pd.DataFrame([fn(ps.preds[(model, s)]) for s in ps.seeds(model)])
    out = rows.mean().to_dict()
    if len(rows) > 1 and "rmse" in rows:
        out["rmse_seed_std"] = float(rows["rmse"].std(ddof=1))
    return out


def error_by_range(ps: PredictionSet, cfg: dict) -> pd.DataFrame:
    """Pooled (window-level) error by glycemic range, per cohort and model. NOTE: pooled over
    windows, so patients with more readings weigh more; this differs from the patient-averaged
    headline RMSE. ``n_readings`` is how many scored windows fall in the range."""
    rng, y, rows = glycemic_range(ps.meta["y"].to_numpy()), ps.meta["y"].to_numpy(), []
    for cohort, cm in _cohort_masks(ps, cfg).items():
        for code, name in enumerate(RANGES):
            m = cm & (rng == code)
            for model in MODELS:
                stats = _seed_avg(ps, model, lambda p, m=m: _err_stats(p[m], y[m])) if m.any() else {}
                rows.append({"horizon_min": ps.horizon_min, "cohort": cohort, "range": name, "model": model,
                             "n_readings": int(m.sum()), "n_patients": int(ps.meta.loc[m, "patient"].nunique()),
                             **stats})
    return pd.DataFrame(rows)


def error_by_range_per_patient(ps: PredictionSet) -> pd.DataFrame:
    """Same, per patient, with ``low_n`` marking ranges with fewer than LOW_N readings."""
    rng, y, rows = glycemic_range(ps.meta["y"].to_numpy()), ps.meta["y"].to_numpy(), []
    for pid in sorted(ps.meta["patient"].unique()):
        pm = (ps.meta["patient"] == pid).to_numpy()
        for code, name in enumerate(RANGES):
            m = pm & (rng == code)
            for model in MODELS:
                stats = _seed_avg(ps, model, lambda p, m=m: _err_stats(p[m], y[m])) if m.any() else {}
                rows.append({"horizon_min": ps.horizon_min, "patient": pid, "range": name, "model": model,
                             "n_readings": int(m.sum()), "low_n": int(m.sum()) < LOW_N, **stats})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Clarke error grid

def clarke_zone(act, pred) -> np.ndarray:
    """Clarke error-grid zone letters (A-E) for reference glucose ``act`` and prediction ``pred``
    (mg/dL), after Clarke et al. (1987).

    Verification: cross-checked against an independent implementation (the ``clarke_error_grid``
    0.1.4 package on PyPI, whose plotted boundary lines carry the same constants: 70, 180, 240, 290,
    the +/-20% lines, +110 upper-C line and the (130, 0)-(180, 70) lower-C line). Over 300,000 random
    continuous points the two agree on every zone; on the integer grid 20-400 x 0-450 they differ
    only for points lying exactly ON a boundary (0.13% of points), because they treat a point on a
    line differently (this code: strict "<" for the 20% band and the <70 rule; theirs: inclusive).
    On our test windows that changes a zone percentage by at most 0.16 percentage points
    (persistence only, whose forecasts are integers); ridge and the LSTMs are unaffected. Not
    checked against Clarke's original paper itself."""
    a, p = np.asarray(act, float), np.asarray(pred, float)
    zone = np.full(a.shape, "B", dtype="<U1")  # default: any remaining point is B (upper or lower)
    decided = np.zeros(a.shape, bool)

    def put(mask, letter):
        nonlocal decided
        m = mask & ~decided
        zone[m] = letter
        decided |= m

    put(((a < 70) & (p < 70)) | (np.abs(a - p) < 0.2 * a), "A")
    put((a <= 70) & (p >= 180), "E")                                   # left-upper
    put((a >= 180) & (p <= 70), "E")                                   # right-lower
    put((a >= 240) & (p >= 70) & (p <= 180), "D")                      # right
    put((a <= 70) & (p >= 70) & (p <= 180), "D")                       # left
    put((a >= 70) & (a <= 290) & (p >= a + 110), "C")                  # upper
    put((a >= 130) & (a <= 180) & (p <= (7 / 5) * a - 182), "C")       # lower
    return zone


def clarke_table(ps: PredictionSet, cfg: dict) -> pd.DataFrame:
    """Percentage of scored windows in each Clarke zone, per cohort and model (seed-averaged)."""
    y, rows = ps.meta["y"].to_numpy(), []
    for cohort, cm in _cohort_masks(ps, cfg).items():
        for model in MODELS:
            pct = _seed_avg(ps, model, lambda p: {f"pct_{z}": 100 * float(np.mean(clarke_zone(y[cm], p[cm]) == z))
                                                  for z in "ABCDE"})
            rows.append({"horizon_min": ps.horizon_min, "cohort": cohort, "model": model, "n_readings": int(cm.sum()),
                         **pct, "pct_A+B": pct["pct_A"] + pct["pct_B"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ lag / anticipation

def observed_lookup(cfg: dict) -> dict[int, tuple[pd.DatetimeIndex, np.ndarray]]:
    """Per test patient: grid index and REAL readings only (NaN where imputed or missing)."""
    patients = [p for ps in cfg["cohorts"].values() for p in ps]
    segs = make_segments(cfg, load_grids(cfg["paths"]["processed_dir"], patients, "test"), "test")
    return {pid: (g.index, g["glucose"].where(g["glucose_observed"]).to_numpy()) for pid, g in segs}


def lagged_actuals(ps: PredictionSet, lookup, horizon_steps: int, n_lags: int) -> np.ndarray:
    """(N, n_lags + 1): actual glucose at target time minus s bins, s = 0..n_lags; NaN if not a
    real reading. Column 0 is the target itself."""
    out = np.full((len(ps.meta), n_lags + 1), np.nan)
    for pid, (index, obs) in lookup.items():
        rows = np.flatnonzero((ps.meta["patient"] == pid).to_numpy())
        pos = index.get_indexer(pd.DatetimeIndex(ps.meta["target_ts"].to_numpy()[rows]))
        for s in range(n_lags + 1):
            ok = pos - s >= 0
            out[rows[ok], s] = obs[pos[ok] - s]
    return out


def lag_and_anticipation(ps: PredictionSet, cfg: dict, lookup) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two questions per model, pooled over windows:

    Lag curve: RMSE between the forecast for time T and the ACTUAL glucose at T - s bins, for
    s = 0..horizon+6, on windows where all those readings are real (same windows for every s and
    model). A forecast that is just the last value shifted forward matches best at s = horizon
    (its own age); one that anticipates matches best nearer s = 0.

    Anticipation: delta_pred = pred - last input, delta_act = actual - last input; correlation,
    OLS slope of delta_pred on delta_act (``delta_slope``) and the share of large moves
    (>= LARGE_MOVE mg/dL) whose direction is right. Persistence has delta_pred = 0.
    CAUTION: ``delta_slope`` (predicted change regressed on the ACTUAL change) is below 1 even for a
    perfectly calibrated forecaster, because conditioning on the outcome selects windows where the
    forecast was more moderate; do not read it as timidity. The calibration check is the reverse
    regression in ``reverse_slope_table``."""
    step = cfg["grid"]["step_min"]
    h = ps.horizon_min // step
    n_lags = h + LAG_EXTRA_STEPS
    lagged = lagged_actuals(ps, lookup, h, n_lags)
    common = ~np.isnan(lagged).any(axis=1)
    y, last = ps.meta["y"].to_numpy(), ps.meta["last"].to_numpy()
    d_act, big = y - last, np.abs(y - last) >= LARGE_MOVE
    curves, rows = [], []
    for cohort, cm in _cohort_masks(ps, cfg).items():
        m = cm & common
        for model in MODELS:
            per_seed = [np.sqrt(np.mean((ps.preds[(model, s)][m, None] - lagged[m]) ** 2, axis=0)) for s in ps.seeds(model)]
            curve = np.mean(per_seed, axis=0)
            best = int(np.argmin(curve))
            best_per_seed = [int(np.argmin(c)) * step for c in per_seed]

            def anticipation(p, cm=cm):
                dp = (p - last)[cm]
                return {"delta_corr": float(np.corrcoef(dp, d_act[cm])[0, 1]) if dp.std() > 0 else 0.0,
                        "delta_slope": float(np.polyfit(d_act[cm], dp, 1)[0]),
                        "direction_acc_large_moves": float(np.mean(np.sign(dp[big[cm]]) == np.sign(d_act[cm][big[cm]])))}

            ant = _seed_avg(ps, model, anticipation)
            rows.append({"horizon_min": ps.horizon_min, "cohort": cohort, "model": model,
                         "n_lag_windows": int(m.sum()), "best_lag_min": best * step, "best_lag_per_seed_min": str(best_per_seed),
                         "rmse_at_lag0": curve[0], "rmse_at_best_lag": curve[best], "rmse_at_horizon_lag": curve[h],
                         "n_large_moves": int((cm & big).sum()), **ant})
            if cohort == "all":
                curves += [{"horizon_min": ps.horizon_min, "model": model, "lag_min": s * step, "rmse": curve[s]}
                           for s in range(n_lags + 1)]
    return pd.DataFrame(rows), pd.DataFrame(curves)


def residual_summary(ps: PredictionSet) -> pd.DataFrame:
    """Bias, spread and tails of pred - actual over all scored windows (seed-averaged)."""
    y, rows = ps.meta["y"].to_numpy(), []
    for model in MODELS:
        s = _seed_avg(ps, model, lambda p: {"bias": float(np.mean(p - y)), "sd": float(np.std(p - y)),
                                             "p5": float(np.percentile(p - y, 5)), "p95": float(np.percentile(p - y, 95))})
        rows.append({"horizon_min": ps.horizon_min, "model": model, "n_readings": len(y), **s})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration and hypo detection

def reverse_slope_table(ps: PredictionSet, cfg: dict) -> pd.DataFrame:
    """Calibration check on changes: regress the ACTUAL change (actual - last input) on the PREDICTED
    change (forecast - last input). Slope 1 with intercept 0 means that, on average, a predicted change
    of d mg/dL is followed by an actual change of d. (The forward slope in the lag table regresses the
    predicted change on the actual change; that slope is below 1 even for a well-calibrated forecaster,
    because conditioning on the actual outcome selects windows where the forecast was too moderate.)
    Persistence predicts no change, so its slope is undefined (NaN). Seed-averaged for the LSTMs."""
    y, last, rows = ps.meta["y"].to_numpy(), ps.meta["last"].to_numpy(), []
    for cohort, cm in _cohort_masks(ps, cfg).items():
        for model in MODELS:
            def fit(p, cm=cm):
                dp, da = (p - last)[cm], (y - last)[cm]
                if dp.std() == 0:
                    return {"reverse_slope": np.nan, "reverse_intercept": np.nan, "corr": np.nan}
                slope, intercept = np.polyfit(dp, da, 1)
                return {"reverse_slope": float(slope), "reverse_intercept": float(intercept),
                        "corr": float(np.corrcoef(dp, da)[0, 1])}
            rows.append({"horizon_min": ps.horizon_min, "cohort": cohort, "model": model, "n_windows": int(cm.sum()),
                         **_seed_avg(ps, model, fit)})
    return pd.DataFrame(rows)


ALERT_THRESHOLDS = (70.0, 80.0, 90.0)  # forecast values below which an alert would be raised (mg/dL)


def hypo_detection(ps: PredictionSet, cfg: dict, thresholds=ALERT_THRESHOLDS) -> pd.DataFrame:
    """How well does "forecast < threshold" identify an actual value below 70 mg/dL?

    The event is always actual < 70 (HYPO); only the alert threshold on the forecast varies.
    sensitivity = P(alert | actual < 70) = TP / (TP + FN); precision = P(actual < 70 | alert) =
    TP / (TP + FP). Counts are windows, pooled over patients (the LSTMs' counts and rates are
    averaged over seeds). For persistence the forecast is the last input value."""
    y, rows = ps.meta["y"].to_numpy(), []
    actual = y < HYPO
    for cohort, cm in _cohort_masks(ps, cfg).items():
        for threshold in thresholds:
            for model in MODELS:
                def stats(p, cm=cm, threshold=threshold):
                    alert = p < threshold
                    tp, fp = int((alert & actual & cm).sum()), int((alert & ~actual & cm).sum())
                    fn = int((~alert & actual & cm).sum())
                    return {"tp": tp, "fp": fp, "fn": fn,
                            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                            "precision": tp / (tp + fp) if tp + fp else np.nan}
                rows.append({"horizon_min": ps.horizon_min, "cohort": cohort, "alert_threshold": threshold,
                             "model": model, "n_actual_hypo": int((actual & cm).sum()), **_seed_avg(ps, model, stats)})
    return pd.DataFrame(rows)


def forecast_spread(ps: PredictionSet) -> pd.DataFrame:
    """Spread of the forecasts versus the spread of the actual target values (all scored windows),
    and how often each falls below 70 mg/dL. A conditional-mean forecast is expected to be less
    spread out than the glucose it predicts; persistence, a real past reading, is not."""
    y = ps.meta["y"].to_numpy()
    rows = []
    for model in MODELS:
        s = _seed_avg(ps, model, lambda p: {"sd_forecast": float(np.std(p)),
                                             "pct_forecast_below_70": 100 * float(np.mean(p < HYPO)),
                                             "pct_forecast_above_180": 100 * float(np.mean(p > HYPER))})
        rows.append({"horizon_min": ps.horizon_min, "model": model, "n_windows": len(y), **s,
                     "sd_actual": float(np.std(y)), "pct_actual_below_70": 100 * float(np.mean(y < HYPO)),
                     "pct_actual_above_180": 100 * float(np.mean(y > HYPER))})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ representative window

def representative_window(ps: PredictionSet, span: int = 96, model: str = INSULIN_LABEL, seed: int = 0):
    """Row indices of an 8 h stretch (``span`` consecutive 5-min targets, all scored) chosen by rule
    so it is not cherry-picked: the patient with the lower-median RMSE of ``model`` (seed ``seed``),
    and within that patient the stretch whose local RMSE is closest to the patient's overall RMSE
    (earliest on ties). Returns (patient, row indices)."""
    err = ps.preds[(model, seed)] - ps.meta["y"].to_numpy()
    per_patient = pd.Series(err**2).groupby(ps.meta["patient"].to_numpy()).mean() ** 0.5
    patient = per_patient.sort_values().index[(len(per_patient) - 1) // 2]
    rows = np.flatnonzero((ps.meta["patient"] == patient).to_numpy())
    rows = rows[np.argsort(ps.meta["target_ts"].to_numpy()[rows])]
    ts = ps.meta["target_ts"].to_numpy()[rows]
    contiguous = np.r_[True, np.diff(ts) == np.timedelta64(5, "m")]
    run_id = np.cumsum(~contiguous)
    best, best_gap = None, np.inf
    for start in range(0, len(rows) - span + 1):
        if run_id[start] != run_id[start + span - 1]:
            continue
        local = float(np.sqrt(np.mean(err[rows[start : start + span]] ** 2)))
        if abs(local - per_patient[patient]) < best_gap - 1e-12:
            best, best_gap = start, abs(local - per_patient[patient])
    if best is None:
        raise ValueError(f"patient {patient} has no {span}-step contiguous stretch of scored windows")
    return int(patient), rows[best : best + span]


# ------------------------------------------------------------------ CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Step-6 evaluation extras.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebuild", action="store_true", help="recompute cached predictions")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = _results_dir(cfg)
    (out / "figures").mkdir(exist_ok=True)
    lookup = observed_lookup(cfg)

    sets = {h: load_or_build(cfg, h, args.rebuild) for h in (6, 12)}
    tables = {"range": [], "range_pp": [], "clarke": [], "lag": [], "curves": [], "resid": [], "calib": [], "hypo": [], "spread": []}
    for ps in sets.values():
        h = ps.horizon_min // cfg["grid"]["step_min"]
        tables["range"].append(error_by_range(ps, cfg))
        tables["range_pp"].append(error_by_range_per_patient(ps))
        tables["clarke"].append(clarke_table(ps, cfg))
        lag, curves = lag_and_anticipation(ps, cfg, lookup)
        tables["lag"].append(lag), tables["curves"].append(curves)
        tables["resid"].append(residual_summary(ps))
        tables["calib"].append(reverse_slope_table(ps, cfg))
        tables["hypo"].append(hypo_detection(ps, cfg))
        tables["spread"].append(forecast_spread(ps))
    files = {"range": "error_by_range", "range_pp": "error_by_range_per_patient", "clarke": "clarke_zones",
             "lag": "lag_and_anticipation", "curves": "lag_curves", "resid": "residual_summary",
             "calib": "calibration_reverse_slope", "hypo": "hypo_detection", "spread": "forecast_spread"}
    result = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    for k, name in files.items():
        result[k].round(4).to_csv(out / f"{name}.csv", index=False)

    from src import plots

    plots.make_figures(cfg, sets, result, out / "figures")
    log.info("wrote %s and figures to %s", ", ".join(f"{n}.csv" for n in files.values()), out)


if __name__ == "__main__":
    main()
