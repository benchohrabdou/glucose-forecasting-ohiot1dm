"""Step-6 analysis on synthetic predictions with known answers (no real patient data)."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    MODELS,
    PredictionSet,
    clarke_table,
    clarke_zone,
    error_by_range,
    error_by_range_per_patient,
    glycemic_range,
    lag_and_anticipation,
    representative_window,
)
from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL

CFG = {"cohorts": {"2018": [1], "2020": [2]}, "grid": {"step_min": 5}}
T0 = pd.Timestamp("2021-12-07")


def test_glycemic_range_boundaries_follow_the_definition():
    y = np.array([40, 69.9, 70, 100, 180, 180.1, 400])
    assert glycemic_range(y).tolist() == [0, 0, 1, 1, 1, 2, 2]


@pytest.mark.parametrize("act,pred,zone", [
    (100, 100, "A"), (100, 119, "A"), (60, 65, "A"),      # within 20%, or both hypoglycemic
    (100, 130, "B"), (100, 70, "B"),                      # outside 20% but harmless
    (60, 200, "E"), (200, 60, "E"),                       # dangerous: hypo read as high, and vice versa
    (250, 120, "D"), (60, 120, "D"),                      # failure to detect hyper / hypo
    (100, 250, "C"), (150, 20, "C"),                      # over-correction
])
def test_clarke_zone_on_hand_worked_points(act, pred, zone):
    assert clarke_zone([act], [pred])[0] == zone


@pytest.mark.parametrize("act,pred,zone", [
    # just either side of each boundary line of the Clarke grid
    (100, 119.9, "A"), (100, 120.1, "B"), (100, 80.1, "A"), (100, 79.9, "B"),   # the +/-20% lines
    (69.9, 69.9, "A"), (100, 69.9, "B"),                                       # both <70 is A; else the 20% rule
    (60, 71.9, "A"), (60, 72.1, "D"), (60, 179.9, "D"), (60, 180.1, "E"),      # left: within 20%, then D up to 180, E above
    (250, 179.9, "D"), (250, 180.1, "B"), (250, 70.1, "D"), (250, 69.9, "E"),  # right: D for pred 70-180, E below 70
    (239, 120, "B"),                                                           # right-D needs act >= 240
    (100, 209.9, "B"), (100, 210.1, "C"), (300, 420, "B"),                     # upper C: pred >= act+110, act <= 290
    (150, 28.1, "B"), (150, 27.9, "C"), (129, 0, "B"),                         # lower C line through (130,0)-(180,70)
    (181, 69.9, "E"), (179, 60, "C"),                                          # right E needs act >= 180
])
def test_clarke_zone_just_either_side_of_each_boundary(act, pred, zone):
    assert clarke_zone([act], [pred])[0] == zone


def test_clarke_matches_the_independent_geometry_on_a_dense_continuous_grid():
    """Zone areas from the independent implementation's boundary lines (70, 180, 240, 290, 20%, +110,
    7/5*act-182) re-derived here point by point; agreement is total away from the lines."""
    rng = np.random.default_rng(0)
    a, p = rng.uniform(20, 400, 50_000), rng.uniform(0, 450, 50_000)

    def by_geometry(x, y):
        if (x < 70 and y < 70) or 0.8 * x < y < 1.2 * x: return "A"
        if (x >= 180 and y <= 70) or (x <= 70 and y >= 180): return "E"
        if (70 <= x <= 290 and y >= x + 110) or (130 <= x <= 180 and y <= 1.4 * x - 182): return "C"
        if (x >= 240 and 70 <= y <= 180) or (x <= 70 and 70 <= y <= 180): return "D"
        return "B"

    assert (clarke_zone(a, p) == np.array([by_geometry(x, y) for x, y in zip(a, p)])).all()


def make_ps(n=40, seeds=(0,)):
    y = np.r_[np.full(10, 50.0), np.full(20, 120.0), np.full(10, 250.0)]
    meta = pd.DataFrame({"patient": np.repeat([1, 2], n // 2), "target_ts": T0 + pd.to_timedelta(np.arange(n) * 5, "m"),
                         "y": y, "last": y, "cohort": np.repeat(["2018", "2020"], n // 2)})
    preds = {("persistence", 0): y.copy(), ("ridge", 0): y + 5.0}
    for s in seeds:
        preds[(GLUCOSE_LABEL, s)] = y + 10.0 * (s + 1)
        preds[(INSULIN_LABEL, s)] = y - 4.0
    return PredictionSet(30, meta, preds)


def test_error_by_range_counts_and_seed_averaging():
    ps = make_ps(seeds=(0, 1))  # glucose-only errors +10 and +20 -> RMSE 15 when averaged over seeds
    d = error_by_range(ps, CFG).query("cohort == 'all'").set_index(["model", "range"])
    assert d.loc[("persistence", "hypo (<70)"), "n_readings"] == 10
    assert d.loc[("persistence", "in range (70-180)"), "n_readings"] == 20
    assert d.loc[("persistence", "hyper (>180)"), "n_readings"] == 10
    assert d.loc[(GLUCOSE_LABEL, "hypo (<70)"), "rmse"] == pytest.approx(15.0)
    assert d.loc[(GLUCOSE_LABEL, "hypo (<70)"), "rmse_seed_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert d.loc[(INSULIN_LABEL, "hyper (>180)"), "bias"] == pytest.approx(-4.0)
    assert d.loc[("ridge", "in range (70-180)"), "mae"] == pytest.approx(5.0)


def test_per_patient_ranges_flag_small_counts():
    d = error_by_range_per_patient(make_ps()).query("model == 'persistence'")
    assert d["low_n"].all()  # every range has < 30 readings in this toy set
    assert d.set_index(["patient", "range"]).loc[(1, "hypo (<70)"), "n_readings"] == 10


def test_clarke_table_percentages_sum_to_100_and_persistence_is_all_a():
    t = clarke_table(make_ps(), CFG).query("cohort == 'all'").set_index("model")
    assert t.loc["persistence", "pct_A"] == pytest.approx(100.0)
    for m in MODELS:
        assert t.loc[m, [f"pct_{z}" for z in "ABCDE"]].sum() == pytest.approx(100.0)


# ---- lag / anticipation on a series with a known answer

H, STEP = 6, 5
N_LAGS = H + 6


def lag_fixture(pred_of):
    """One patient with a smooth glucose trace; pred_of(g, pos) builds a forecast for target `pos`."""
    g = 150 + 60 * np.sin(np.arange(400) / 11.0)
    index = pd.date_range(T0, periods=len(g), freq="5min")
    pos = np.arange(N_LAGS + 12, len(g))
    y, last = g[pos], g[pos - H]
    pred = pred_of(g, pos)
    meta = pd.DataFrame({"patient": 1, "target_ts": index[pos], "y": y, "last": last, "cohort": "2018"})
    ps = PredictionSet(H * STEP, meta, {(m, 0): pred.copy() for m in MODELS})
    return ps, {1: (index, g)}


def test_repeat_last_value_matches_best_at_its_own_age_and_an_oracle_at_lag_zero():
    ps, lookup = lag_fixture(lambda g, pos: g[pos - H])            # persistence-like forecast
    table, curves = lag_and_anticipation(ps, {**CFG, "cohorts": {"2018": [1]}}, lookup)
    row = table.query("cohort == 'all' and model == 'persistence'").iloc[0]
    assert row["best_lag_min"] == H * STEP and row["rmse_at_best_lag"] == pytest.approx(0, abs=1e-9)
    assert row["delta_slope"] == pytest.approx(0, abs=1e-9) and row["direction_acc_large_moves"] == 0

    ps, lookup = lag_fixture(lambda g, pos: g[pos])                # perfect forecast
    table, _ = lag_and_anticipation(ps, {**CFG, "cohorts": {"2018": [1]}}, lookup)
    row = table.query("cohort == 'all' and model == 'persistence'").iloc[0]
    assert row["best_lag_min"] == 0 and row["rmse_at_lag0"] == pytest.approx(0, abs=1e-9)
    assert row["delta_corr"] == pytest.approx(1) and row["delta_slope"] == pytest.approx(1)
    assert row["direction_acc_large_moves"] == pytest.approx(1)
    assert row["n_large_moves"] > 0 and row["n_lag_windows"] > 100


def test_lag_windows_are_restricted_to_windows_with_real_readings_at_every_lag():
    ps, lookup = lag_fixture(lambda g, pos: g[pos - H])
    index, g = lookup[1]
    g = g.copy()
    g[200:203] = np.nan  # an outage: a window is unusable if any of its lags 0..12 lands on it
    table, _ = lag_and_anticipation(ps, {**CFG, "cohorts": {"2018": [1]}}, {1: (index, g)})
    n = table.query("cohort == 'all' and model == 'persistence'")["n_lag_windows"].iloc[0]
    assert n == len(ps.meta) - (3 + N_LAGS)  # targets 200..214 are dropped


# ---- representative window

def test_representative_window_is_rule_based_contiguous_and_scored():
    n = 300
    rng = np.random.default_rng(0)
    y = 120 + rng.normal(0, 20, 3 * n)
    meta = pd.DataFrame({"patient": np.repeat([1, 2, 3], n), "target_ts": np.tile(T0 + pd.to_timedelta(np.arange(n) * 5, "m"), 3),
                         "y": y, "last": y, "cohort": "2018"})
    scale = np.repeat([5.0, 10.0, 20.0], n)                        # per-patient error scale -> patient 2 is the median
    pred = y + rng.normal(0, 1, 3 * n) * scale
    ps = PredictionSet(30, meta, {(INSULIN_LABEL, 0): pred})
    patient, rows = representative_window(ps)
    assert patient == 2 and len(rows) == 96
    ts = meta["target_ts"].to_numpy()[rows]
    assert (np.diff(ts) == np.timedelta64(5, "m")).all()
    assert (meta["patient"].to_numpy()[rows] == 2).all()
    with pytest.raises(ValueError):
        representative_window(ps, span=400)


# ---- figures render (smoke test: files exist and are non-trivial PNGs)

def test_all_figures_render(tmp_path):
    from src import plots

    def big_ps(minutes):
        n = 300
        rng = np.random.default_rng(minutes)
        y = 130 + 40 * np.sin(np.arange(2 * n) / 15) + rng.normal(0, 5, 2 * n)
        meta = pd.DataFrame({"patient": np.repeat([1, 2], n), "target_ts": np.tile(T0 + pd.to_timedelta(np.arange(n) * 5, "m"), 2),
                             "y": y, "last": y - 3, "cohort": np.repeat(["2018", "2020"], n)})
        preds = {(m, 0): y + rng.normal(0, 8 + i, 2 * n) for i, m in enumerate(MODELS)}
        return PredictionSet(minutes, meta, preds)

    sets = {6: big_ps(30), 12: big_ps(60)}
    ranges = pd.concat([error_by_range(ps, CFG) for ps in sets.values()])
    ps, lookup = lag_fixture(lambda g, pos: g[pos - H])
    lag, curves = lag_and_anticipation(ps, {**CFG, "cohorts": {"2018": [1]}}, lookup)
    plots.make_figures(CFG, sets, {"range": ranges, "curves": curves, "lag": lag}, tmp_path)
    for name in ("predicted_vs_actual", "residual_distributions", "error_by_range", "lag_curves"):
        f = tmp_path / f"{name}.png"
        assert f.exists() and f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and f.stat().st_size > 5_000
