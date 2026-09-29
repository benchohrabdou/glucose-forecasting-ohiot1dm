"""Probabilistic forecasts for hypoglycemia alerting (quantile LSTM, and a direct classifier).

Same windows, splits, scaling and rejection rules as the MSE models; only the head and the loss
differ. Nothing here touches the existing MSE results or checkpoints.

    python -m src.probabilistic train  --configs configs/lstm_q_ph30.yaml ... --seeds 0 1 2 3 4
    python -m src.probabilistic report --config configs/base.yaml

Protocol, fixed before any test result was seen:
  * quantile LSTM: pinball loss averaged over quantiles; early stopping on validation mean pinball
    loss (mg/dL). Alert when q_tau < 70 mg/dL, tau chosen on VALIDATION from TAU_CANDIDATES.
  * persistence and MSE LSTM: alert when the forecast < T, T chosen on VALIDATION from THRESHOLDS.
  * classifier: class-weighted BCE (positive weight = negatives / positives in training windows);
    early stopping on validation weighted BCE. Alert when P(glucose < 70) >= p, p chosen on
    VALIDATION from PROB_THRESHOLDS.
  * Selection criterion: F2 on validation, pooled over patients, averaged over the five seeds;
    one rule per model family and horizon. Ties go to the option with fewer validation alerts, then
    to the first option in the grid. Test files are scored once, afterwards.
"""
from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import Scaler, build_datasets, collect_arrays, feature_columns
from src.evaluate import (GLUCOSE_LABEL, INSULIN_LABEL, _results_dir, cohort_of, load_model, predict,
                          verify_fingerprint)
from src.models import build_model
from src.utils import get_logger, load_config, set_seed

log = get_logger(__name__)

HYPO = 70.0
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
TAU_CANDIDATES = (0.05, 0.10, 0.25, 0.50)
THRESHOLDS = tuple(float(t) for t in range(70, 111, 5))
PROB_THRESHOLDS = tuple(round(0.01 * i, 2) for i in range(1, 100))
INTERVALS = {"80%": (0.10, 0.90), "90%": (0.05, 0.95)}
RANGES = {"all": None, "hypo (<70)": (-np.inf, 70.0), "in range (70-180)": (70.0, 180.0), "hyper (>180)": (180.0, np.inf)}
SEEDS = (0, 1, 2, 3, 4)

Q_GLU, Q_INS = "quantile lstm (glucose only)", "quantile lstm (glucose + insulin/carbs)"
C_GLU, C_INS = "classifier lstm (glucose only)", "classifier lstm (glucose + insulin/carbs)"
# model family -> (config stem template, kind)
FAMILIES = {
    GLUCOSE_LABEL: ("lstm_ph{m}", "mse"), INSULIN_LABEL: ("lstm_ins_ph{m}", "mse"),
    Q_GLU: ("lstm_q_ph{m}", "quantile"), Q_INS: ("lstm_ins_q_ph{m}", "quantile"),
    C_GLU: ("lstm_cls_ph{m}", "classifier"), C_INS: ("lstm_ins_cls_ph{m}", "classifier"),
}


# ------------------------------------------------------------------ losses and metrics

def pinball_loss(pred: torch.Tensor, y: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """Mean over batch and quantiles of max(tau * e, (tau - 1) * e), e = y - q. pred (B, K), y (B, 1)."""
    e = y - pred
    return torch.maximum(taus * e, (taus - 1) * e).mean()


def positive_weight(labels: np.ndarray) -> float:
    """Negatives / positives, the BCE weight that balances the two classes."""
    pos = int(np.sum(labels))
    if pos == 0:
        raise ValueError("no positive (hypoglycemic) training windows")
    return (len(labels) - pos) / pos


def detection(alert: np.ndarray, actual: np.ndarray) -> dict:
    """Counts and rates for an alert vector against actual lows. Precision is NaN with no alerts;
    F2 is 0 when there is no true positive (sensitivity is then 0)."""
    alert, actual = np.asarray(alert, bool), np.asarray(actual, bool)
    tp, fp, fn = int((alert & actual).sum()), int((alert & ~actual).sum()), int((~alert & actual).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f2 = 5 * prec * sens / (4 * prec + sens) if tp > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "alerts": tp + fp, "sensitivity": sens, "precision": prec, "f2": f2}


def choose_rule(options: dict, actual: np.ndarray) -> tuple[object, pd.DataFrame]:
    """options: {param: [alert vector per seed]} on VALIDATION. Returns the param with the highest
    seed-averaged F2 (ties: fewer mean alerts, then grid order) and the full selection table."""
    rows = []
    for order, (param, per_seed) in enumerate(options.items()):
        d = pd.DataFrame([detection(a, actual) for a in per_seed])
        rows.append({"param": param, "order": order, "val_f2": d["f2"].mean(), "val_sensitivity": d["sensitivity"].mean(),
                     "val_precision": d["precision"].mean(), "val_alerts": d["alerts"].mean(),
                     "per_seed_f2": ",".join(f"{v:.3f}" for v in d["f2"])})
    table = pd.DataFrame(rows)
    best = table.sort_values(["val_f2", "val_alerts", "order"], ascending=[False, True, True]).iloc[0]
    table["selected"] = table["order"] == best["order"]
    return best["param"], table.drop(columns="order")


def interval_stats(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> dict:
    """Empirical coverage of [lo, hi] and its mean width (mg/dL)."""
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo))}


def range_mask(y: np.ndarray, name: str) -> np.ndarray:
    """Actual-value range masks; in range is 70-180 inclusive, as in the rest of the repo."""
    if name == "all":
        return np.ones(len(y), bool)
    if name == "hypo (<70)":
        return y < 70
    if name == "hyper (>180)":
        return y > 180
    return (y >= 70) & (y <= 180)


# ------------------------------------------------------------------ training

def _labels(y_scaled: torch.Tensor, scaler: Scaler) -> torch.Tensor:
    return (torch.as_tensor(scaler.unscale_glucose(y_scaled.numpy())) < HYPO).float()


def train_prob(cfg: dict, name: str) -> Path:
    """Train a quantile or classifier LSTM; identical optimiser, clipping, patience and seeding to
    src.train. Early stopping on validation mean pinball loss (mg/dL) or validation weighted BCE."""
    seed, tcfg, kind = cfg["seed"], cfg["train"], cfg["model"]["type"]
    set_seed(seed)
    train_ds, val_ds, _, scaler = build_datasets(cfg)
    n_features = len(feature_columns(cfg))
    xva, yva, _, _ = collect_arrays(val_ds)
    model = build_model(cfg, n_features)
    optimiser = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    extra = {}
    if kind == "lstm_quantile":
        taus = torch.tensor(model.quantiles, dtype=torch.float32)
        loss_fn = lambda out, y: pinball_loss(out, y, taus)  # noqa: E731
        val_metric = lambda: float(pinball_loss(predict(model, xva), yva, taus)) * scaler.std["glucose"]  # noqa: E731
        metric_name = "val_pinball_mgdl"
    elif kind == "lstm_classifier":
        _, ytr, _, _ = collect_arrays(train_ds)
        pw = positive_weight(_labels(ytr, scaler).numpy().ravel())
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
        loss_fn = lambda out, y: bce(out, _labels(y, scaler))  # noqa: E731
        yva_lab = _labels(yva, scaler)
        val_metric = lambda: float(bce(predict(model, xva), yva_lab))  # noqa: E731
        metric_name, extra = "val_weighted_bce", {"pos_weight": pw}
    else:
        raise ValueError(f"train_prob does not handle model type {kind!r}")

    results_dir, ckpt_dir = _results_dir(cfg), Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}_resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log.info("%s: seed=%d, %d train / %d val windows %s", name, seed, len(train_ds), len(val_ds), extra)

    best, best_state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(1, tcfg["max_epochs"] + 1):
        model.train()
        losses = []
        for x, y, _ in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimiser.step()
            losses.append(loss.item())
        v = val_metric()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), metric_name: v})
        log.info("epoch %3d  train_loss=%.4f  %s=%.4f", epoch, history[-1]["train_loss"], metric_name, v)
        if v < best:
            best, best_epoch, best_state = v, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tcfg["patience"]:
            log.info("early stop: no validation improvement for %d epochs", tcfg["patience"])
            break

    pd.DataFrame(history).to_csv(results_dir / f"{name}_training_log.csv", index=False)
    path = ckpt_dir / f"{name}.pt"
    torch.save({"model_state": best_state, "cfg": cfg, "scaler": {"mean": scaler.mean, "std": scaler.std},
                "columns": feature_columns(cfg), "seed": seed, "best_epoch": best_epoch, metric_name: best, **extra}, path)
    scaler.save(ckpt_dir / f"{name}.scaler.json")
    log.info("best epoch %d, %s %.4f -> %s", best_epoch, metric_name, best, path)
    return path


# ------------------------------------------------------------------ predictions

def _outputs(model, scaler: Scaler, x: torch.Tensor, kind: str) -> np.ndarray:
    out = predict(model, x).numpy()
    if kind == "classifier":
        return 1 / (1 + np.exp(-out.ravel()))
    unscaled = scaler.unscale_glucose(out)
    return unscaled if kind == "quantile" else unscaled.ravel()


def build_predictions(cfg: dict, horizon: int, config_dir: str | Path = "configs", seeds=SEEDS) -> dict:
    """Validation and test outputs of persistence and every available family/seed at one horizon.

    Returns {"horizon_min", "val": meta, "test": meta, "preds": {(family, seed, split): array}}.
    meta has patient, y (mg/dL), last (last input, mg/dL), cohort. Every checkpoint's test windows
    are checked against the baselines' fingerprint and must equal the reference windows exactly."""
    minutes = horizon * cfg["grid"]["step_min"]
    ref_cfg = load_config(Path(config_dir) / f"lstm_ph{minutes}.yaml")
    _, val, test, scaler = build_datasets(ref_cfg)
    cohorts = cohort_of(cfg)
    out = {"horizon_min": minutes, "preds": {}}
    ref = {}
    for split, ds in (("val", val), ("test", test)):
        x, y, p, t = collect_arrays(ds)
        if split == "test":
            verify_fingerprint(ref_cfg, minutes, p, t)
        ref[split] = (p, t)
        out[split] = pd.DataFrame({"patient": p, "target_ts": t, "y": scaler.unscale_glucose(y.numpy()).ravel(),
                                   "last": scaler.unscale_glucose(x[:, -1, 0].numpy()), "cohort": [cohorts[int(i)] for i in p]})
        out["preds"][("persistence", 0, split)] = out[split]["last"].to_numpy()

    for family, (stem, kind) in FAMILIES.items():
        fcfg = load_config(Path(config_dir) / (stem.format(m=minutes) + ".yaml"))
        for seed in seeds:
            ckpt = Path(fcfg["paths"]["checkpoint_dir"]) / f"{stem.format(m=minutes)}_seed{seed}.pt"
            if not ckpt.exists():
                continue
            if kind == "mse":
                model, ck_scaler, _ = load_model(fcfg, ckpt, len(feature_columns(fcfg)))
            else:
                ck = torch.load(ckpt, map_location="cpu", weights_only=True)
                for key in ("window", "features", "model", "gap", "split"):
                    if fcfg[key] != ck["cfg"][key]:
                        raise ValueError(f"{ckpt.name}: config section '{key}' differs from training")
                model = build_model(fcfg, len(feature_columns(fcfg)))
                model.load_state_dict(ck["model_state"])
                ck_scaler = Scaler(**ck["scaler"])
            _, fval, ftest, _ = build_datasets(fcfg, scaler=ck_scaler)
            for split, ds in (("val", fval), ("test", ftest)):
                x, _, p, t = collect_arrays(ds)
                if not (np.array_equal(p, ref[split][0]) and np.array_equal(t, ref[split][1])):
                    raise AssertionError(f"{ckpt.name}: {split} windows differ from the reference windows")
                if split == "test":
                    verify_fingerprint(fcfg, minutes, p, t)
                out["preds"][(family, seed, split)] = _outputs(model, ck_scaler, x, kind)
    return out


def load_or_build(cfg: dict, horizon: int, rebuild: bool = False) -> dict:
    """Cached in <processed_dir> (patient-level, gitignored)."""
    path = Path(cfg["paths"]["processed_dir"]) / f"prob_predictions_ph{horizon * cfg['grid']['step_min']}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    d = build_predictions(cfg, horizon)
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return d


def seeds_of(d: dict, family: str) -> list[int]:
    return sorted({s for (f, s, sp) in d["preds"] if f == family and sp == "test"})


# ------------------------------------------------------------------ reports

def alert_vector(family: str, kind: str, output: np.ndarray, param) -> np.ndarray:
    if kind == "quantile":
        return output[:, QUANTILES.index(param)] < HYPO
    if kind == "classifier":
        return output >= param
    return output < param


def _grid(kind: str):
    return {"quantile": TAU_CANDIDATES, "classifier": PROB_THRESHOLDS}.get(kind, THRESHOLDS)


def alerting_tables(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(validation selection table, test table) for persistence and every family present."""
    families = [("persistence", "mse")] + [(f, k) for f, (_, k) in FAMILIES.items() if seeds_of(d, f)]
    val_low, test_low = d["val"]["y"].to_numpy() < HYPO, d["test"]["y"].to_numpy() < HYPO
    sel_rows, test_rows = [], []
    for family, kind in families:
        seeds = [0] if family == "persistence" else seeds_of(d, family)
        options = {p: [alert_vector(family, kind, d["preds"][(family, s, "val")], p) for s in seeds] for p in _grid(kind)}
        best, table = choose_rule(options, val_low)
        per_seed_best = [choose_rule({p: [v[i]] for p, v in options.items()}, val_low)[0] for i in range(len(seeds))]
        table.insert(0, "model", family)
        table.insert(1, "horizon_min", d["horizon_min"])
        table.insert(2, "rule", {"quantile": "q_tau < 70", "classifier": "P(<70) >= p"}.get(kind, "forecast < T"))
        table["n_val_lows"] = int(val_low.sum())
        sel_rows.append(table)
        tst = pd.DataFrame([detection(alert_vector(family, kind, d["preds"][(family, s, "test")], best), test_low) for s in seeds])
        test_rows.append({"horizon_min": d["horizon_min"], "model": family, "rule": table["rule"].iloc[0], "chosen_param": best,
                          "per_seed_best_param_on_val": ",".join(str(b) for b in per_seed_best), "n_seeds": len(seeds),
                          "n_val_lows": int(val_low.sum()), "n_test_lows": int(test_low.sum()),
                          "sensitivity": tst["sensitivity"].mean(), "precision": tst["precision"].mean(),
                          "precision_seeds_with_alerts": int(tst["precision"].notna().sum()),
                          "f2": tst["f2"].mean(), "f2_seed_std": tst["f2"].std(ddof=1) if len(seeds) > 1 else np.nan,
                          "alerts": tst["alerts"].mean(), "tp": tst["tp"].mean(), "fp": tst["fp"].mean()})
    return pd.concat(sel_rows, ignore_index=True), pd.DataFrame(test_rows)


def classifier_scores(d: dict) -> pd.DataFrame:
    """Threshold-free scores on test: PR-AUC (average precision) and Brier score, seed-averaged."""
    from sklearn.metrics import average_precision_score

    y = (d["test"]["y"].to_numpy() < HYPO).astype(int)
    rows = []
    for family in (C_GLU, C_INS):
        seeds = seeds_of(d, family)
        if not seeds:
            continue
        pr = [average_precision_score(y, d["preds"][(family, s, "test")]) for s in seeds]
        br = [float(np.mean((d["preds"][(family, s, "test")] - y) ** 2)) for s in seeds]
        rows.append({"horizon_min": d["horizon_min"], "model": family, "n_seeds": len(seeds), "n_test_lows": int(y.sum()),
                     "prevalence": float(y.mean()), "pr_auc": float(np.mean(pr)), "pr_auc_seed_std": float(np.std(pr, ddof=1)) if len(pr) > 1 else np.nan,
                     "brier": float(np.mean(br)), "brier_of_constant_prevalence": float(y.mean() * (1 - y.mean()))})
    return pd.DataFrame(rows)


def calibration_table(d: dict) -> pd.DataFrame:
    """Coverage and mean width of the 80% and 90% intervals, overall and by actual range (test)."""
    y, rows = d["test"]["y"].to_numpy(), []
    for family in (Q_GLU, Q_INS):
        seeds = seeds_of(d, family)
        for rng in RANGES:
            m = range_mask(y, rng)
            for iv, (lo, hi) in INTERVALS.items():
                per = [interval_stats(q[m, QUANTILES.index(lo)], q[m, QUANTILES.index(hi)], y[m])
                       for q in (d["preds"][(family, s, "test")] for s in seeds)]
                p = pd.DataFrame(per)
                rows.append({"horizon_min": d["horizon_min"], "model": family, "range": rng, "interval": iv,
                             "nominal": hi - lo, "n_readings": int(m.sum()), "coverage": p["coverage"].mean(),
                             "coverage_seed_std": p["coverage"].std(ddof=1), "mean_width_mgdl": p["mean_width"].mean()})
    return pd.DataFrame(rows)


def point_accuracy(d: dict) -> pd.DataFrame:
    """Per-patient RMSE / MAE of the median (quantile LSTM) and of the MSE LSTM, averaged over seeds."""
    y, pat, rows = d["test"]["y"].to_numpy(), d["test"]["patient"].to_numpy(), []
    for family in (GLUCOSE_LABEL, INSULIN_LABEL, Q_GLU, Q_INS):
        for seed in seeds_of(d, family):
            out = d["preds"][(family, seed, "test")]
            point = out[:, QUANTILES.index(0.5)] if out.ndim == 2 else out
            for p in np.unique(pat):
                m = pat == p
                e = point[m] - y[m]
                rows.append({"model": family, "horizon_min": d["horizon_min"], "patient": int(p), "seed": seed,
                             "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))})
    per_seed = pd.DataFrame(rows)
    return per_seed.groupby(["model", "horizon_min", "patient"], sort=False)[["rmse", "mae"]].mean().reset_index()


def summarize_point(per_patient: pd.DataFrame, cohorts: dict) -> pd.DataFrame:
    rows = []
    pp = per_patient.assign(cohort=per_patient["patient"].map(cohorts))
    for cohort in ("all", "2020", "2018"):
        sub = pp if cohort == "all" else pp[pp["cohort"] == cohort]
        for (model, h), g in sub.groupby(["model", "horizon_min"], sort=False):
            rows.append({"cohort": cohort, "model": model, "horizon_min": h, "n_patients": len(g),
                         "rmse_mean": g["rmse"].mean(), "rmse_std": g["rmse"].std(ddof=1),
                         "mae_mean": g["mae"].mean(), "mae_std": g["mae"].std(ddof=1)})
    return pd.DataFrame(rows)


def report(cfg: dict, rebuild: bool = False) -> dict:
    out = _results_dir(cfg)
    tables = {k: [] for k in ("selection", "alerting", "classifier", "calibration", "point_pp")}
    for h in (6, 12):
        d = load_or_build(cfg, h, rebuild)
        sel, tst = alerting_tables(d)
        tables["selection"].append(sel), tables["alerting"].append(tst)
        tables["classifier"].append(classifier_scores(d))
        tables["calibration"].append(calibration_table(d))
        tables["point_pp"].append(point_accuracy(d))
    res = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    res["point"] = summarize_point(res["point_pp"], cohort_of(cfg))
    files = {"selection": "quantile_alert_selection_validation", "alerting": "quantile_alerting_test",
             "classifier": "classifier_scores_test", "calibration": "quantile_calibration_test",
             "point_pp": "quantile_point_accuracy_per_patient", "point": "quantile_point_accuracy_summary"}
    for k, name in files.items():
        res[k].round(4).to_csv(out / f"{name}.csv", index=False)
    from src import plots_prob

    plots_prob.make_figures(res, out / "figures")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantile / classifier LSTMs for hypoglycemia alerting.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--configs", nargs="+", required=True)
    t.add_argument("--seeds", type=int, nargs="+", required=True)
    r = sub.add_parser("report")
    r.add_argument("--config", required=True)
    r.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.cmd == "train":
        for seed in args.seeds:
            for config in args.configs:
                cfg, name = load_config(config), f"{Path(config).stem}_seed{seed}"
                cfg["seed"] = seed
                if (Path(cfg["paths"]["checkpoint_dir"]) / f"{name}.pt").exists():
                    log.info("%s already trained, skipping", name)
                    continue
                train_prob(cfg, name)
    else:
        report(load_config(args.config), args.rebuild)


if __name__ == "__main__":
    main()
