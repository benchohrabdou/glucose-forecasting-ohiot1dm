"""Windowing, rejection, scaling and leakage tests on synthetic grids (no real patient data)."""
import random

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.data.dataset import (
    GlucoseWindowDataset,
    Scaler,
    chronological_split,
    feature_columns,
    make_segments,
    valid_window_ends,
)
from src.data.preprocess import apply_gap_policy

STEP = pd.Timedelta(minutes=5)
T0 = pd.Timestamp("2021-12-07 00:00:00")
CFG = {
    "features": {"insulin_carbs": False, "time_of_day": False},
    "gap": {"max_interp_gap_min": 30},
    "grid": {"step_min": 5},
    "split": {"val_frac": 0.2},
}


def make_grid(n=200, missing=(), seed=0) -> pd.DataFrame:
    """Un-filled grid: glucose NaN at `missing` bins; observed marks real readings."""
    rng = np.random.default_rng(seed)
    glucose = 120 + 30 * np.sin(np.arange(n) / 9) + rng.normal(0, 2, n)
    glucose[list(missing)] = np.nan
    return pd.DataFrame(
        {
            "glucose": glucose,
            "glucose_observed": ~np.isnan(glucose),
            "bolus": rng.random(n),
            "basal": np.full(n, 0.1),
            "carbs": np.zeros(n),
        },
        index=pd.date_range(T0, periods=n, freq="5min"),
    )


def dataset(grid, L=12, PH=6, frac=0.25, columns=None, scaler=None, filled=True):
    seg = [(1, apply_gap_policy(grid, 30) if filled else grid)]
    scaler = scaler or Scaler.fit(seg)
    return GlucoseWindowDataset(seg, scaler, columns or feature_columns(CFG), L, PH, frac), scaler


# ------------------------------------------------------------- spec: timestamps

@pytest.mark.parametrize("L,PH", [(12, 6), (12, 12), (24, 6), (24, 12)])
def test_target_is_exactly_horizon_after_last_input_and_no_input_at_or_after_target(L, PH):
    ds, _ = dataset(make_grid(400, missing=[50, 51, 120, *range(200, 212)]), L, PH)
    assert len(ds) > 50
    for i in random.Random(0).sample(range(len(ds)), 50):
        _, _, meta = ds[i]
        ts = ds.timestamps(i)
        target = pd.Timestamp(meta["target_ts"])
        assert len(ts) == L
        assert target - ts[-1] == PH * STEP
        assert pd.Timestamp(meta["input_end_ts"]) == ts[-1]
        assert (ts < target).all()


def test_y_is_the_real_reading_at_target_in_mgdl_after_unscaling():
    grid = make_grid(120, missing=[40])
    ds, scaler = dataset(grid)
    for i in range(len(ds)):
        _, y, meta = ds[i]
        assert scaler.unscale_glucose(y.item()) == pytest.approx(grid.loc[pd.Timestamp(meta["target_ts"]), "glucose"], abs=1e-3)


# ------------------------------------------------------------- rejection rules

def test_target_must_be_a_real_reading():
    grid = make_grid(60, missing=[30])
    _, counts = valid_window_ends(*_arrays(grid), 12, 6, 0.25)
    assert counts["candidates"] == 60 - 6 - 11
    assert counts["rejected_target_not_observed"] == 1  # only t = 24 targets bin 30
    assert counts["kept"] == counts["candidates"] - 1  # one imputed input bin (1/12) is allowed


def test_imputed_fraction_boundary_three_of_twelve_kept_four_rejected():
    grid = make_grid(60, missing=[20, 21, 22, 23])  # 4 forward-filled bins
    _, counts = valid_window_ends(*_arrays(grid), 12, 6, 0.25)
    assert counts["rejected_target_not_observed"] == 4  # t = 14..17
    assert counts["rejected_imputed_frac"] == 9  # windows holding all 4 imputed bins: t = 23..31
    keep3, _ = valid_window_ends(*_arrays(make_grid(60, missing=[20, 21, 22])), 12, 6, 0.25)
    assert 22 in keep3 and 25 in keep3  # windows with exactly 3/12 = 0.25 imputed pass


def test_windows_with_input_nan_after_fill_are_rejected_and_never_emitted():
    grid = make_grid(120, missing=range(40, 50))  # 10-bin gap: 6 filled, 4 stay NaN
    _, counts = valid_window_ends(*_arrays(grid), 12, 6, 1.0)  # frac rule off to isolate NaN rule
    assert counts["rejected_nan_input"] > 0
    ds, _ = dataset(grid, frac=1.0)
    for i in range(len(ds)):
        x, y, _ = ds[i]
        assert not np.isnan(x.numpy()).any() and not np.isnan(y.numpy()).any()


def _arrays(grid):
    filled = apply_gap_policy(grid, 30)
    return filled["glucose"].to_numpy(), filled["glucose_observed"].to_numpy()


# ------------------------------------------------------------- split boundary

def test_no_window_straddles_the_train_val_boundary():
    L, PH = 12, 6
    grid = make_grid(300)
    grids = {1: grid}
    train, val = make_segments(CFG, grids, "train"), make_segments(CFG, grids, "val")
    cut = chronological_split(grid, 0.2)[1].index[0]
    scaler = Scaler.fit(train)
    dtr = GlucoseWindowDataset(train, scaler, feature_columns(CFG), L, PH, 0.25)
    dva = GlucoseWindowDataset(val, scaler, feature_columns(CFG), L, PH, 0.25)
    for i in range(len(dtr)):
        assert pd.Timestamp(dtr[i][2]["target_ts"]) < cut  # train target never in validation
    for i in range(len(dva)):
        assert dva.timestamps(i)[0] >= cut  # validation input never reaches back into training
    full, _ = valid_window_ends(*_arrays(grid), L, PH, 0.25)
    assert len(dtr) + len(dva) == len(full) - (L + PH - 1)  # exactly the straddlers are gone


def test_split_is_chronological_and_unshuffled():
    grid = make_grid(100)
    tr, va = chronological_split(grid, 0.2)
    assert tr.index.is_monotonic_increasing and va.index.is_monotonic_increasing
    assert tr.index[-1] < va.index[0] and len(tr) == 80 and len(va) == 20


def test_glucose_is_refilled_per_segment_not_across_the_cut():
    grid = make_grid(200, missing=[160, 161])  # val_frac 0.2 -> cut at bin 160
    filled_whole = apply_gap_policy(grid, 30)
    assert filled_whole["glucose"].iloc[160:162].notna().all()  # would be filled from training
    (_, val), = make_segments(CFG, {1: grid}, "val")
    assert val["glucose"].iloc[:2].isna().all()  # per-segment refill: nothing crosses the cut


def test_make_segments_fills_short_gaps_causally_for_every_part():
    grid = make_grid(200, missing=[50, 51])
    for part in ("train", "val", "test"):
        (_, seg), = make_segments(CFG, {1: grid.copy()}, part)
        if part == "train":
            assert (seg["glucose"].iloc[50:52] == seg["glucose"].iloc[49]).all()  # carried forward
            assert seg["glucose_observed"].iloc[50:52].tolist() == [False, False]
        assert seg["glucose"].notna().sum() >= seg["glucose_observed"].sum()


# ------------------------------------------------------------- scaling

def test_scaler_uses_training_segment_only_and_real_readings_only():
    grid = make_grid(300, missing=[10, 11, 12])
    tampered = grid.copy()
    tampered.iloc[240:, tampered.columns.get_loc("glucose")] += 500  # inside validation
    s1 = Scaler.fit(make_segments(CFG, {1: grid}, "train"))
    s2 = Scaler.fit(make_segments(CFG, {1: tampered}, "train"))
    assert s1 == s2  # validation values cannot influence the statistics
    tr = chronological_split(grid, 0.2)[0]
    assert s1.mean["glucose"] == pytest.approx(tr.loc[tr["glucose_observed"], "glucose"].mean())


def test_scaler_roundtrip_and_persistence(tmp_path):
    scaler = Scaler.fit(make_segments(CFG, {1: make_grid(200)}, "train"))
    scaler.save(tmp_path / "scaler.json")
    assert Scaler.load(tmp_path / "scaler.json") == scaler
    assert scaler.unscale_glucose(scaler.scale_glucose(137.0)) == pytest.approx(137.0)


# ------------------------------------------------------------- features / causality

def test_glucose_observed_is_a_model_input_and_flags_imputed_bins():
    assert feature_columns(CFG) == ["glucose", "glucose_observed"]
    assert feature_columns({"features": {"insulin_carbs": True, "time_of_day": True}}) == [
        "glucose", "glucose_observed", "bolus", "basal", "carbs", "tod_sin", "tod_cos"]
    grid = make_grid(100, missing=[40])
    ds, _ = dataset(grid)
    hit = 0
    for i in range(len(ds)):
        x, _, _ = ds[i]
        ts = ds.timestamps(i)
        expected = grid.loc[ts, "glucose_observed"].to_numpy(dtype=np.float32)
        assert np.array_equal(x[:, 1].numpy(), expected)
        hit += int(expected.min() == 0)
    assert hit > 0  # some windows really contain the imputed bin


def test_later_readings_cannot_change_earlier_window_inputs():
    grid = make_grid(200, missing=[70, 71, 72])
    changed = grid.copy()
    changed.iloc[90, changed.columns.get_loc("glucose")] += 80  # a reading at bin 90
    scaler = Scaler.fit([(1, apply_gap_policy(grid, 30))])
    a, _ = dataset(grid, scaler=scaler)
    b, _ = dataset(changed, scaler=scaler)
    cutoff = grid.index[90].value
    checked = 0
    for i in range(len(a)):
        xa, _, meta = a[i]
        if meta["input_end_ts"] < cutoff:
            assert np.array_equal(xa.numpy(), b[i][0].numpy())
            checked += 1
    assert checked > 20


def test_insulin_carb_columns_are_scaled_with_train_statistics():
    cfg = {**CFG, "features": {"insulin_carbs": True, "time_of_day": False}}
    grid = make_grid(200)
    seg = make_segments(cfg, {1: grid}, "train")
    scaler = Scaler.fit(seg)
    x = scaler.transform(seg[0][1], feature_columns(cfg))
    assert x.shape[1] == 5
    assert x[:, 2].mean() == pytest.approx(0, abs=1e-5)  # bolus standardised


def test_dataloader_collates_meta():
    ds, _ = dataset(make_grid(100))
    x, y, meta = next(iter(DataLoader(ds, batch_size=8)))
    assert x.shape == (8, 12, 2) and y.shape == (8, 1)
    assert meta["patient"].shape == (8,) and meta["target_ts"].dtype.is_floating_point is False


# ------------------------------------------------------------- build_datasets end to end

def _write_cache(tmp_path, val_shift=0.0):
    proc = tmp_path / "processed"
    proc.mkdir()
    train = make_grid(300, seed=1)
    train.iloc[240:, train.columns.get_loc("glucose")] += val_shift  # inside the validation part
    train.to_pickle(proc / "1_train.pkl")
    make_grid(120, seed=2).to_pickle(proc / "1_test.pkl")
    return {**CFG, "cohorts": {"2018": [1]}, "window": {"window_len": 12, "horizon": 6, "max_imputed_frac": 0.25},
            "paths": {"processed_dir": str(proc), "results_dir": str(tmp_path / "results")}}


def test_build_datasets_fits_scaler_on_training_part_only_and_reports_boundary(tmp_path):
    from src.data.dataset import build_datasets

    tmp_a, tmp_b = tmp_path / "a", tmp_path / "b"
    tmp_a.mkdir()
    tmp_b.mkdir()
    _, _, _, clean = build_datasets(_write_cache(tmp_a))
    train, val, test, tampered = build_datasets(_write_cache(tmp_b, val_shift=500.0))
    assert tampered == clean  # a +500 mg/dL shift confined to validation cannot move the scaler
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    counts = pd.read_csv(tmp_b / "results" / "window_counts_ph30.csv")
    assert set(counts["split"]) == {"train", "val", "test"}
    assert counts.loc[counts["split"] == "val", "boundary_dropped"].item() == 12 + 6 - 1
