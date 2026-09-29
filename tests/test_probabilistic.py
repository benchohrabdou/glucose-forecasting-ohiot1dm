"""Quantile / classifier LSTMs and the validation-only alert-rule selection (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm import LSTMClassifier, QuantileLSTMForecaster
from src.probabilistic import (
    QUANTILES,
    choose_rule,
    detection,
    interval_stats,
    pinball_loss,
    positive_weight,
    range_mask,
    train_prob,
)
from test_evaluate import full_cfg


def test_quantile_head_is_ordered_by_construction_and_keeps_the_lstm_body():
    torch.manual_seed(0)
    m = QuantileLSTMForecaster(3, list(QUANTILES), hidden_size=8, num_layers=2)
    with torch.no_grad():
        m.head.bias.fill_(-5.0)                       # large negative raw gaps: softplus still >= 0
        out = m(torch.randn(200, 12, 3) * 10)
    assert out.shape == (200, 7) and (out.diff(dim=1) >= 0).all()
    assert {k.split(".")[0] for k in m.state_dict()} == {"lstm", "head"}
    with pytest.raises(ValueError):
        QuantileLSTMForecaster(3, [0.1, 0.9])      # no median
    with pytest.raises(ValueError):
        QuantileLSTMForecaster(3, [0.9, 0.5, 0.1])  # not increasing


def test_median_column_is_the_raw_median_output():
    torch.manual_seed(1)
    m = QuantileLSTMForecaster(2, [0.1, 0.5, 0.9], hidden_size=4, num_layers=1)
    x = torch.randn(5, 12, 2)
    m.eval()
    with torch.no_grad():
        _, (h, _) = m.lstm(x)
        raw = m.head(m.drop(h[-1]))
        assert torch.allclose(m(x)[:, 1], raw[:, 1])


def test_classifier_outputs_one_logit():
    assert LSTMClassifier(3, hidden_size=8)(torch.randn(4, 12, 3)).shape == (4, 1)


def test_pinball_loss_by_hand_and_minimised_at_the_empirical_quantile():
    taus = torch.tensor([0.1, 0.9])
    pred, y = torch.tensor([[0.0, 0.0]]), torch.tensor([[2.0]])     # e = 2 for both
    assert pinball_loss(pred, y, taus).item() == pytest.approx((0.1 * 2 + 0.9 * 2) / 2)
    pred, y = torch.tensor([[4.0, 4.0]]), torch.tensor([[2.0]])     # e = -2
    assert pinball_loss(pred, y, taus).item() == pytest.approx((0.9 * 2 + 0.1 * 2) / 2)
    ys = torch.arange(1.0, 101.0).unsqueeze(1)                       # 1..100
    grid = torch.arange(1.0, 101.0)
    losses = [pinball_loss(torch.full((100, 1), float(c)), ys, torch.tensor([0.9])).item() for c in grid]
    assert grid[int(np.argmin(losses))] in (90.0, 91.0)


def test_detection_and_f2_by_hand():
    d = detection([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])                   # tp 2, fp 1, fn 1
    assert (d["tp"], d["fp"], d["fn"], d["alerts"]) == (2, 1, 1, 3)
    p, r = 2 / 3, 2 / 3
    assert d["f2"] == pytest.approx(5 * p * r / (4 * p + r))
    none = detection([0, 0], [1, 0])
    assert np.isnan(none["precision"]) and none["f2"] == 0 and none["sensitivity"] == 0


def test_choose_rule_uses_seed_averaged_f2_and_breaks_ties_by_fewer_alerts():
    actual = np.array([1, 1, 0, 0, 0, 0], bool)
    options = {
        "a": [np.array([1, 0, 0, 0, 0, 0], bool)] * 2,              # f2 = 5*1*.5/(4+.5)
        "b": [np.array([1, 1, 1, 1, 1, 1], bool)] * 2,              # sens 1, prec 1/3
        "c": [np.array([1, 1, 1, 0, 0, 0], bool), np.array([1, 1, 0, 0, 0, 0], bool)],
    }
    best, table = choose_rule(options, actual)
    assert best == "c" and table.set_index("param").loc["c", "selected"]
    tied = {"many": [np.array([1, 1, 1, 0, 0, 0], bool)], "few": [np.array([1, 1, 1, 0, 0, 0], bool)]}
    tied["few"] = [np.array([1, 1, 1, 0, 0, 0], bool)]
    assert choose_rule(tied, actual)[0] == "many"                     # full tie -> grid order
    tied2 = {"x": [np.array([1, 0, 0, 0, 0, 0], bool)], "y": [np.array([1, 0, 0, 0, 0, 0], bool)]}
    assert choose_rule(tied2, actual)[0] == "x"


def test_interval_stats_and_range_masks():
    y = np.array([60.0, 100.0, 200.0, 70.0, 180.0])
    s = interval_stats(np.array([50, 90, 210, 60, 170.0]), np.array([65, 110, 230, 80, 190.0]), y)
    assert s["coverage"] == pytest.approx(4 / 5) and s["mean_width"] == pytest.approx((15 + 20 + 20 + 20 + 20) / 5)
    assert range_mask(y, "hypo (<70)").tolist() == [True, False, False, False, False]
    assert range_mask(y, "in range (70-180)").tolist() == [False, True, False, True, True]
    assert range_mask(y, "hyper (>180)").sum() == 1 and range_mask(y, "all").all()


def test_positive_weight_is_negative_over_positive():
    assert positive_weight(np.array([1, 0, 0, 0])) == 3
    with pytest.raises(ValueError):
        positive_weight(np.zeros(5))


def _prob_cfg(tmp_path, model):
    cfg = full_cfg(tmp_path)
    cfg["model"] = {"hidden_size": 8, "num_layers": 1, "dropout": 0.1, **model}
    return cfg


def test_train_quantile_smoke_saves_checkpoint_with_ordered_outputs(tmp_path):
    cfg = _prob_cfg(tmp_path, {"type": "lstm_quantile", "quantiles": list(QUANTILES)})
    path = train_prob(cfg, "q_smoke")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["seed"] == 0 and "val_pinball_mgdl" in ck and set(ck["scaler"]) == {"mean", "std"}
    log = pd.read_csv(tmp_path / "results" / "q_smoke_training_log.csv")
    assert "val_pinball_mgdl" in log and len(log) >= 1


def test_train_classifier_smoke_records_the_positive_weight(tmp_path):
    cfg = _prob_cfg(tmp_path, {"type": "lstm_classifier"})
    # the synthetic grid oscillates around 120 +/- 30 mg/dL: shift it so some targets are < 70
    for f in (tmp_path / "processed").glob("*.pkl"):
        g = pd.read_pickle(f)
        g["glucose"] -= 45
        g.to_pickle(f)
    path = train_prob(cfg, "c_smoke")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert ck["pos_weight"] > 1 and "val_weighted_bce" in ck


# ---------------------------------------------------------------- reporting on synthetic predictions

from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL  # noqa: E402
from src.probabilistic import (  # noqa: E402
    C_GLU,
    C_INS,
    Q_GLU,
    Q_INS,
    alerting_tables,
    calibration_table,
    classifier_scores,
    point_accuracy,
    summarize_point,
)


def synthetic_set(n=400, seeds=(0, 1), seed=0):
    """Prediction dict shaped like build_predictions' output, with known behaviour:
    persistence = truth + 8 (alerts need a raised threshold), MSE LSTMs = truth shrunk toward 120,
    quantiles = truth + fixed offsets (so the 10% quantile is 20 below the truth),
    classifier = 0.9 for lows, 0.1 otherwise."""
    rng = np.random.default_rng(seed)
    d = {"horizon_min": 30, "preds": {}}
    offsets = np.array([-30, -20, -10, 0, 10, 20, 30], float)
    for split in ("val", "test"):
        y = rng.uniform(40, 250, n)
        d[split] = pd.DataFrame({"patient": np.repeat([1, 2], n // 2), "target_ts": np.arange(n), "y": y,
                                 "last": y + 8, "cohort": np.repeat(["2018", "2020"], n // 2)})
        d["preds"][("persistence", 0, split)] = y + 8
        for s in seeds:
            for fam in (GLUCOSE_LABEL, INSULIN_LABEL):
                d["preds"][(fam, s, split)] = 120 + 0.5 * (y - 120)
            for fam in (Q_GLU, Q_INS):
                d["preds"][(fam, s, split)] = y[:, None] + offsets[None, :]
            for fam in (C_GLU, C_INS):
                d["preds"][(fam, s, split)] = np.where(y < 70, 0.9, 0.1)
    return d


def test_alerting_tables_select_on_validation_and_report_test():
    d = synthetic_set()
    sel, tst = alerting_tables(d)
    t = tst.set_index("model")
    # alert iff truth + 8 < T: T = 75 catches only truth < 67 (misses lows); T = 80 catches all
    # (truth < 72) with a few false alarms, which F2 prefers because it weights sensitivity
    assert t.loc["persistence", "chosen_param"] == 80.0
    assert t.loc["persistence", "sensitivity"] == 1 and t.loc["persistence", "precision"] < 1
    assert t.loc[Q_GLU, "chosen_param"] == 0.5                        # median == truth: perfect alerts
    assert t.loc[Q_GLU, "sensitivity"] == 1 and t.loc[Q_GLU, "precision"] == 1
    assert t.loc[C_INS, "f2"] == pytest.approx(1.0)
    assert (tst["n_val_lows"] > 0).all() and (tst["n_test_lows"] > 0).all()
    assert sel.groupby("model")["selected"].sum().eq(1).all()        # exactly one chosen rule per model
    assert set(sel.loc[sel["model"] == Q_GLU, "param"]) == {0.05, 0.10, 0.25, 0.50}


def test_selection_never_looks_at_test_data():
    d = synthetic_set()
    before = alerting_tables(d)[1].set_index("model")["chosen_param"]
    for key in list(d["preds"]):
        if key[2] == "test":
            d["preds"][key] = np.zeros_like(d["preds"][key])             # wreck every test prediction
    after = alerting_tables(d)[1].set_index("model")["chosen_param"]
    assert before.equals(after)


def test_calibration_table_on_known_offsets():
    c = calibration_table(synthetic_set())
    row = c[(c["model"] == Q_GLU) & (c["range"] == "all") & (c["interval"] == "80%")].iloc[0]
    assert row["coverage"] == 1 and row["mean_width_mgdl"] == pytest.approx(40)   # [-20, +20] always covers
    assert set(c["range"]) == {"all", "hypo (<70)", "in range (70-180)", "hyper (>180)"}
    assert (c.groupby(["model", "interval"])["n_readings"].first() == 400).all()


def test_point_accuracy_uses_the_median_and_averages_seeds():
    pp = point_accuracy(synthetic_set())
    assert pp[pp["model"] == Q_GLU]["rmse"].max() == pytest.approx(0)
    mse = pp[pp["model"] == GLUCOSE_LABEL]
    assert (mse["rmse"] > 0).all() and len(mse) == 2
    s = summarize_point(pp, {1: "2018", 2: "2020"})
    assert set(s["cohort"]) == {"all", "2018", "2020"}


def test_classifier_scores_perfect_ranking():
    s = classifier_scores(synthetic_set()).set_index("model")
    assert s.loc[C_GLU, "pr_auc"] == pytest.approx(1.0)
    assert s.loc[C_GLU, "brier"] == pytest.approx(0.01)
    assert 0 < s.loc[C_GLU, "prevalence"] < 1


def test_figures_render(tmp_path):
    from src import plots_prob

    d = synthetic_set()
    _, tst = alerting_tables(d)
    res = {"calibration": calibration_table(d), "alerting": pd.concat([tst, tst.assign(horizon_min=60)])}
    res["calibration"] = pd.concat([res["calibration"], res["calibration"].assign(horizon_min=60)])
    plots_prob.make_figures(res, tmp_path)
    for f in ("quantile_coverage_by_range.png", "alerting_comparison.png"):
        assert (tmp_path / f).read_bytes()[:4] == b"\x89PNG" and (tmp_path / f).stat().st_size > 5000


# ---------------------------------------------------------------- end to end: build_predictions

def test_build_predictions_end_to_end_verifies_windows(tmp_path):
    import yaml

    from src.data.dataset import build_datasets, collect_arrays
    from src.evaluate import write_fingerprint
    from src.probabilistic import build_predictions
    from src.train import train

    base = full_cfg(tmp_path)
    for f in (tmp_path / "processed").glob("*.pkl"):          # push some targets below 70
        g = pd.read_pickle(f)
        g["glucose"] -= 45
        g.to_pickle(f)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    body = {"hidden_size": 8, "num_layers": 1, "dropout": 0.1}
    specs = {"lstm_ph30": ({"type": "lstm", **body}, False), "lstm_ins_ph30": ({"type": "lstm", **body}, True),
             "lstm_q_ph30": ({"type": "lstm_quantile", "quantiles": list(QUANTILES), **body}, False),
             "lstm_ins_q_ph30": ({"type": "lstm_quantile", "quantiles": list(QUANTILES), **body}, True),
             "lstm_cls_ph30": ({"type": "lstm_classifier", **body}, False),
             "lstm_ins_cls_ph30": ({"type": "lstm_classifier", **body}, True)}
    for stem, (model, ins) in specs.items():
        c = {**base, "model": model, "features": {"insulin_carbs": ins, "time_of_day": False}}
        (cfg_dir / f"{stem}.yaml").write_text(yaml.safe_dump(c))
        c["seed"] = 0
        (train if model["type"] == "lstm" else train_prob)(c, f"{stem}_seed0")
    _, _, test, _ = build_datasets(base)
    _, _, p, t = collect_arrays(test)
    write_fingerprint(base, 30, p, t)

    d = build_predictions(base, 6, config_dir=cfg_dir, seeds=(0,))
    assert len(d["preds"]) == 2 + 6 * 2                         # persistence + 6 families, val and test
    q = d["preds"][(Q_INS, 0, "test")]
    assert q.shape == (len(d["test"]), 7) and (np.diff(q, axis=1) >= 0).all()
    pr = d["preds"][(C_GLU, 0, "val")]
    assert ((pr >= 0) & (pr <= 1)).all()
    np.testing.assert_allclose(d["preds"][("persistence", 0, "test")], d["test"]["last"])
