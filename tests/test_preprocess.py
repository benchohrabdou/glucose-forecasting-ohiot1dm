"""Grid construction and gap-policy tests on synthetic tables (no real patient data)."""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    apply_gap_policy,
    build_grid,
    ffill_causal,
)

T0 = pd.Timestamp("2021-12-07 00:00:00")


def ts(minutes: float, seconds: float = 0) -> pd.Timestamp:
    return T0 + pd.Timedelta(minutes=minutes, seconds=seconds)


def tables(glucose, bolus=None, meal=None, basal=None, temp_basal=None):
    empty = pd.DataFrame()
    return {
        "glucose_level": pd.DataFrame(glucose, columns=["ts", "value"]),
        "bolus": bolus if bolus is not None else empty,
        "meal": meal if meal is not None else empty,
        "basal": basal if basal is not None else pd.DataFrame({"ts": [ts(-60)], "value": [1.2]}),
        "temp_basal": temp_basal if temp_basal is not None else empty,
    }


def steady_glucose(n=12, start=100.0):
    return [(ts(5 * i), start + i) for i in range(n)]


def test_snap_jitter_and_mask_and_span():
    g = [(ts(0, 3), 100), (ts(5, -4), 101), (ts(20, 2), 104)]  # 10- and 15-min bins missing
    grid, stats = build_grid(tables(g))
    assert list(grid.index) == [ts(5 * i) for i in range(5)]  # first..last CGM, 5-min steps
    assert grid["glucose_observed"].tolist() == [True, True, False, False, True]
    assert grid["glucose"].isna().tolist() == [False, False, True, True, False]
    assert stats["cgm_collisions"] == 0


def test_collision_keeps_first_and_counts():
    g = [(ts(0), 100), (ts(4), 111), (ts(5.5), 999), (ts(10), 102)]  # 4 and 5.5 both -> bin 5
    grid, stats = build_grid(tables(g))
    assert stats["cgm_collisions"] == 1
    assert grid.loc[ts(5), "glucose"] == 111


def test_instant_bolus_and_carbs_land_in_bin_and_absence_is_zero():
    bolus = pd.DataFrame({"ts_begin": [ts(12)], "ts_end": [ts(12)], "type": ["normal"], "dose": [4.2]})
    meal = pd.DataFrame({"ts": [ts(11)], "type": ["Lunch"], "carbs": [45.0]})
    grid, _ = build_grid(tables(steady_glucose(), bolus=bolus, meal=meal))
    assert grid.loc[ts(10), "bolus"] == pytest.approx(4.2)
    assert grid.loc[ts(10), "carbs"] == 45
    assert grid["bolus"].sum() == pytest.approx(4.2)
    assert grid["carbs"].sum() == 45


def test_extended_bolus_spread_uniformly_and_conserves_dose():
    bolus = pd.DataFrame({"ts_begin": [ts(10)], "ts_end": [ts(30)], "type": ["square"], "dose": [5.0]})
    grid, _ = build_grid(tables(steady_glucose(), bolus=bolus))
    covered = grid.loc[ts(10) : ts(30), "bolus"]
    assert len(covered) == 5 and np.allclose(covered, 1.0)
    assert grid["bolus"].sum() == pytest.approx(5.0)


def test_events_outside_cgm_span_are_dropped_and_counted():
    meal = pd.DataFrame({"ts": [ts(-30), ts(20)], "type": ["Snack"] * 2, "carbs": [10.0, 20.0]})
    bolus = pd.DataFrame({"ts_begin": [ts(500)], "ts_end": [ts(500)], "type": ["normal"], "dose": [1.0]})
    grid, stats = build_grid(tables(steady_glucose(), meal=meal, bolus=bolus))
    assert stats["meal_outside_grid"] == 1 and stats["bolus_outside_grid"] == 1
    assert grid["carbs"].sum() == 20 and grid["bolus"].sum() == 0


def test_basal_is_rate_over_12_and_standing_rate_predates_grid():
    grid, stats = build_grid(tables(steady_glucose()))  # rate 1.2 U/hr set an hour before the grid
    assert np.allclose(grid["basal"], 1.2 / 12)
    assert stats["basal_unknown_bins"] == 0


def test_temp_basal_overrides_and_suspension_is_zero():
    tb = pd.DataFrame({"ts_begin": [ts(10)], "ts_end": [ts(25)], "value": [0.0]})
    grid, _ = build_grid(tables(steady_glucose(), temp_basal=tb))
    assert (grid.loc[ts(10) : ts(20), "basal"] == 0).all()  # [begin, end): 10, 15, 20
    assert grid.loc[ts(25), "basal"] == pytest.approx(0.1)  # back to standing rate
    assert grid.loc[ts(5), "basal"] == pytest.approx(0.1)


def test_unknown_leading_basal_uses_initial_rate_else_zero():
    basal = pd.DataFrame({"ts": [ts(20)], "value": [2.4]})  # first event after the grid starts
    grid0, s0 = build_grid(tables(steady_glucose(), basal=basal))
    gridc, sc = build_grid(tables(steady_glucose(), basal=basal), initial_basal=0.6)
    assert s0["basal_unknown_bins"] == sc["basal_unknown_bins"] == 4
    assert (grid0.loc[: ts(15), "basal"] == 0).all()
    assert np.allclose(gridc.loc[: ts(15), "basal"], 0.05)
    assert grid0.loc[ts(20), "basal"] == pytest.approx(0.2)


def test_time_of_day_is_config_gated():
    off, _ = build_grid(tables(steady_glucose()))
    on, _ = build_grid(tables(steady_glucose()), time_of_day=True)
    assert "tod_sin" not in off and {"tod_sin", "tod_cos"} <= set(on)


# ---------------------------------------------------------------- gap policy

NAN = np.nan


def test_causal_fill_never_uses_future_values():
    base = pd.Series([100, NAN, NAN, 130.0, 140.0])
    perturbed = base.copy()
    perturbed.iloc[3] = 999.0  # change a future reading
    a, b = ffill_causal(base, 6), ffill_causal(perturbed, 6)
    assert a.iloc[:3].tolist() == b.iloc[:3].tolist() == [100, 100, 100]  # carried forward, not interpolated


def test_causal_fill_respects_limit_and_leaves_rest_nan():
    s = pd.Series([100, *([NAN] * 8), 150.0])
    out = ffill_causal(s, 6)
    assert (out.iloc[1:7] == 100).all() and out.iloc[7:9].isna().all()


def test_apply_gap_policy_is_causal_for_every_split_and_keeps_observed_mask():
    g = [(ts(0), 100), (ts(15), 130), (ts(20), 131)]
    grid, _ = build_grid(tables(g))
    filled = apply_gap_policy(grid, 30)
    assert filled.loc[ts(5), "glucose"] == 100 and filled.loc[ts(10), "glucose"] == 100  # carried, not 110/120
    assert filled["glucose_observed"].tolist() == [True, False, False, True, True]  # scoring mask untouched


def test_apply_gap_policy_is_idempotent_and_refills_from_observed_only():
    g = [(ts(0), 100), (ts(15), 130), (ts(20), 131)]
    grid, _ = build_grid(tables(g))
    once = apply_gap_policy(grid, 30)
    assert apply_gap_policy(once, 30).equals(once)
    # Cutting the grid after the first reading must NOT keep values filled from before the cut.
    tail = apply_gap_policy(once.iloc[1:], 30)
    assert tail["glucose"].iloc[:2].isna().all()


def test_long_gap_is_filled_only_up_to_the_limit():
    g = [(ts(0), 100), (ts(60), 150)]  # 11 missing bins
    grid, _ = build_grid(tables(g))
    out = apply_gap_policy(grid, 30)["glucose"]
    assert (out.iloc[1:7] == 100).all() and out.iloc[7:12].isna().all() and out.iloc[12] == 150
