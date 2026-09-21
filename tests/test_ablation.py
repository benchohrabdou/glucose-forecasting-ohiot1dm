"""Multi-seed aggregation and the paired insulin/carbs ablation, against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import GLUCOSE_LABEL, INSULIN_LABEL, write_comparison

BASE = {1: 20.0, 2: 22.0, 3: 24.0}           # patient 1 is 2018; 2 and 3 are 2020
INS_SHIFT = {1: -2.0, 2: +1.0, 3: -0.5}      # insulin/carbs vs glucose-only seed-mean RMSE
GLU_SEED_OFFSET = {0: 0.0, 1: +1.0, 2: -1.0}  # glucose-only varies by seed; insulin/carbs does not


def _write(results, name, model, per_patient, seed):
    rows = [{"model": model, "horizon_min": 30, "patient": p, "n_windows": 100, "rmse": r, "mae": r / 2, "seed": seed}
            for p, r in per_patient.items()]
    pd.DataFrame(rows).to_csv(results / f"model_{name}_seed{seed}_per_patient.csv", index=False)


@pytest.fixture
def cfg(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    pd.DataFrame([{"model": "persistence", "horizon_min": 30, "patient": p, "n_windows": 100,
                   "rmse": r + 5, "mae": 1.0, "seed": 42} for p, r in BASE.items()]) \
        .to_csv(results / "baselines_per_patient.csv", index=False)
    for seed, off in GLU_SEED_OFFSET.items():
        _write(results, "lstm_ph30", GLUCOSE_LABEL, {p: r + off for p, r in BASE.items()}, seed)
        _write(results, "lstm_ins_ph30", INSULIN_LABEL, {p: r + INS_SHIFT[p] for p, r in BASE.items()}, seed)
    # a single-run file from an earlier stage must not leak into the multi-seed tables
    _write(results, "lstm_ph30_old", GLUCOSE_LABEL, {p: 999.0 for p in BASE}, 42)
    (results / "model_lstm_ph30_old_seed42_per_patient.csv").rename(results / "model_lstm_ph30_per_patient.csv")
    return {"cohorts": {"2018": [1], "2020": [2, 3]}, "paths": {"results_dir": str(results)}}


def test_paired_counts_are_from_seed_averaged_rmse(cfg):
    write_comparison(cfg)
    ab = pd.read_csv(f"{cfg['paths']['results_dir']}/ablation_insulin_carbs.csv").set_index("cohort")
    a = ab.loc["all"]
    assert (a["n_patients"], a["n_seeds"], a["n_improved"]) == (3, 3, 2)  # patients 1 and 3 improve
    assert a["mean_delta_rmse"] == pytest.approx(-0.5)
    assert ab.loc["2020", "n_improved"] == 1 and ab.loc["2020", "n_patients"] == 2
    assert ab.loc["2020", "mean_delta_rmse"] == pytest.approx(0.25)
    assert ab.loc["2018", "n_improved"] == 1 and ab.loc["2018", "n_patients"] == 1


def test_per_seed_columns_show_whether_the_result_holds_seed_by_seed(cfg):
    write_comparison(cfg)
    a = pd.read_csv(f"{cfg['paths']['results_dir']}/ablation_insulin_carbs.csv").set_index("cohort").loc["all"]
    # per-seed cross-patient mean differences: -0.5, -1.5, +0.5
    assert a["delta_seed_mean"] == pytest.approx(-0.5) and a["delta_seed_std"] == pytest.approx(1.0)
    assert a["improved_per_seed"] == "2,2,1"


def test_summary_reports_std_across_seeds_and_across_patients_separately(cfg):
    s = write_comparison(cfg)
    g = s[(s["model"] == GLUCOSE_LABEL) & (s["cohort"] == "all")].iloc[0]
    assert g["n_seeds"] == 3
    assert g["rmse_mean"] == pytest.approx(22.0)                       # mean of seed-averaged patient RMSEs
    assert g["rmse_std"] == pytest.approx(np.std([20, 22, 24], ddof=1))  # spread across patients
    assert g["rmse_seed_mean"] == pytest.approx(22.0)                  # per-seed means: 22, 23, 21
    assert g["rmse_seed_std"] == pytest.approx(1.0)                    # spread across seeds
    p = s[(s["model"] == "persistence") & (s["cohort"] == "all")].iloc[0]
    assert p["n_seeds"] == 1 and pd.isna(p["rmse_seed_std"])          # deterministic baseline: no seed spread


def test_files_without_a_seed_suffix_and_unpaired_seeds_are_ignored(cfg, tmp_path):
    s = write_comparison(cfg)
    assert 999.0 not in s["rmse_mean"].round(1).tolist()  # the stray single-run file was not aggregated
    # add a seed only the glucose-only model has: it must not enter the paired comparison
    _write(tmp_path / "results", "lstm_ph30", GLUCOSE_LABEL, {p: 50.0 for p in BASE}, 7)
    write_comparison(cfg)
    ab = pd.read_csv(tmp_path / "results" / "ablation_insulin_carbs.csv").set_index("cohort").loc["all"]
    assert ab["n_seeds"] == 3 and ab["mean_delta_rmse"] == pytest.approx(-0.5)


def test_per_patient_table_is_seed_averaged_with_cohort(cfg):
    write_comparison(cfg)
    t = pd.read_csv(f"{cfg['paths']['results_dir']}/comparison_per_patient.csv")
    row = t[(t["model"] == INSULIN_LABEL) & (t["patient"] == 3)].iloc[0]
    assert row["rmse"] == pytest.approx(23.5) and row["n_seeds"] == 3 and str(row["cohort"]) == "2020"


def test_per_patient_ablation_table_joins_logging_density(cfg):
    out = cfg["paths"]["results_dir"]
    rows = [{"patient": p, "split": sp, "meals_per_day": 3.0 + p, "boluses_per_day": 6.0 + p + (sp == "test")}
            for p in BASE for sp in ("train", "test")]
    pd.DataFrame(rows).to_csv(f"{out}/data_quality.csv", index=False)
    write_comparison(cfg)
    t = pd.read_csv(f"{out}/ablation_per_patient.csv").set_index("patient")
    assert t.loc[3, "delta_rmse"] == pytest.approx(-0.5) and bool(t.loc[3, "improved"])
    assert t.loc[2, "delta_rmse"] == pytest.approx(1.0) and not bool(t.loc[2, "improved"])
    assert t.loc[1, "meals_per_day_train"] == 4.0 and t.loc[1, "boluses_per_day_test"] == 8.0


def test_per_patient_ablation_is_skipped_without_the_event_count_columns(cfg):
    pd.DataFrame([{"patient": 1, "split": "train", "carbs_g_per_day": 1.0}])         .to_csv(f"{cfg['paths']['results_dir']}/data_quality.csv", index=False)
    write_comparison(cfg)  # old-style report: must not crash
