"""Comparison guarantees: same windows, coverage accounting, config inheritance, train->evaluate."""
import copy

import pandas as pd
import pytest
import torch
import yaml

from src.data.dataset import build_datasets, collect_arrays
from src.evaluate import (
    evaluate_checkpoint,
    verify_fingerprint,
    write_comparison,
    write_coverage,
    write_fingerprint,
)
from src.models.lstm import LSTMForecaster
from src.train import train
from src.utils import load_config
from test_dataset import CFG, _write_cache, make_grid


def full_cfg(tmp_path, **feature_overrides):
    cfg = _write_cache(tmp_path)
    cfg["window"]["horizon"] = 6
    cfg["seed"] = 0
    cfg["cohorts"] = {"2018": [1]}
    cfg["features"] = {**CFG["features"], **feature_overrides}
    cfg["model"] = {"type": "lstm", "hidden_size": 8, "num_layers": 1, "dropout": 0.1}
    cfg["train"] = {"lr": 1e-3, "batch_size": 32, "max_epochs": 2, "patience": 5, "grad_clip": 1.0}
    cfg["paths"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    return cfg


def test_test_windows_do_not_depend_on_the_feature_set(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = build_datasets(full_cfg(tmp_path / "a"))[2]
    b = build_datasets(full_cfg(tmp_path / "b", insulin_carbs=True))[2]
    _, _, pa, ta = collect_arrays(a)
    _, _, pb, tb = collect_arrays(b)
    assert (pa == pb).all() and (ta == tb).all()  # insulin/carbs cannot change which windows are scored


def test_fingerprint_verify_accepts_same_windows_and_rejects_others(tmp_path):
    cfg = full_cfg(tmp_path)
    _, _, p, t = collect_arrays(build_datasets(cfg)[2])
    with pytest.raises(FileNotFoundError):
        verify_fingerprint(cfg, 30, p, t)
    write_fingerprint(cfg, 30, p, t)
    verify_fingerprint(cfg, 30, p, t)
    with pytest.raises(AssertionError):
        verify_fingerprint(cfg, 30, p[:-1], t[:-1])  # one window fewer
    with pytest.raises(AssertionError):
        verify_fingerprint(cfg, 30, p, t[::-1].copy())  # same windows, different order
    with pytest.raises(ValueError):
        verify_fingerprint(cfg, 60, p, t)  # no baseline for this horizon


def test_coverage_accounts_for_every_reading_and_matches_the_dataset(tmp_path):
    cfg = full_cfg(tmp_path)
    cov = write_coverage(cfg, (6,))
    n_test = len(build_datasets(cfg)[2])
    assert cov["scored"].sum() == n_test
    row = cov.iloc[0]
    assert row["scored"] + row["unscored_no_history"] + row["unscored_input_gap"] == row["test_cgm_readings"]
    assert row["unscored_no_history"] == 12 - 1 + 6  # first window_len - 1 + horizon readings


def test_load_config_inherits_and_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"window": {"window_len": 12, "horizon": 6}, "seed": 1}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"base": "base.yaml", "window": {"horizon": 12}}))
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"window": {"window_len": 12, "horizon": 12}, "seed": 1}


def test_lstm_interface_single_and_double_layer():
    for layers in (1, 2):
        out = LSTMForecaster(3, hidden_size=8, num_layers=layers)(torch.randn(5, 12, 3))
        assert out.shape == (5, 1)


def test_train_saves_scaler_and_evaluation_reuses_it_on_baseline_windows(tmp_path):
    cfg = full_cfg(tmp_path)
    _, _, p, t = collect_arrays(build_datasets(cfg)[2])
    write_fingerprint(cfg, 30, p, t)
    path = train(cfg, "smoke")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert set(ckpt["scaler"]) == {"mean", "std"} and ckpt["seed"] == 0 and ckpt["columns"] == ["glucose", "glucose_observed"]
    assert (tmp_path / "ckpt" / "smoke.scaler.json").exists()
    assert (tmp_path / "results" / "smoke_training_log.csv").exists()
    assert (tmp_path / "results" / "smoke_resolved_config.yaml").exists()

    table = evaluate_checkpoint(cfg, path, "smoke")
    assert table["n_windows"].sum() == len(p) and table["rmse"].gt(0).all()

    bad = copy.deepcopy(cfg)
    bad["model"]["hidden_size"] = 16
    with pytest.raises(ValueError):
        evaluate_checkpoint(bad, path, "smoke")  # config drifted from the trained checkpoint

    pd.DataFrame([{"model": "persistence", "horizon_min": 30, "patient": 1, "n_windows": 1, "rmse": 1.0, "mae": 1.0, "seed": 0}]) \
        .to_csv(tmp_path / "results" / "baselines_per_patient.csv", index=False)
    summary = write_comparison(cfg)
    assert {"all", "2018"} == set(summary["cohort"]) and (summary["published_bglp_rmse"].fillna("") == "").all()
