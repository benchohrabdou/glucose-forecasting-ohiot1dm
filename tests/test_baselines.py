"""Baseline correctness on hand-checkable inputs."""
import numpy as np
import pytest
import torch

from src.models.baselines import LinearExtrapolation, Persistence, RidgeRegression
from src.utils import per_patient_metrics, rmse, summarize_patients


def window(values):
    """(1, L, 2) window: column 0 glucose, column 1 a dummy flag."""
    v = torch.tensor(values, dtype=torch.float32)
    return torch.stack([v, torch.ones_like(v)], dim=1).unsqueeze(0)


def test_persistence_returns_last_glucose_value():
    assert Persistence()(window([1, 2, 5])).item() == 5


def test_linear_extrapolation_follows_a_perfect_ramp():
    # +2 per step, last value 10, horizon 6 steps -> 10 + 12
    assert LinearExtrapolation(horizon=6, k=4)(window([0, 1, 1, 2, 4, 6, 8, 10])).item() == pytest.approx(22)
    assert LinearExtrapolation(horizon=6, k=2)(window([3, 9, 10])).item() == pytest.approx(10 + 1 * 6)


def test_linear_extrapolation_of_flat_window_is_persistence():
    x = window([7.0] * 12)
    assert LinearExtrapolation(6, 3)(x).item() == Persistence()(x).item() == 7.0


def test_ridge_recovers_a_linear_target_and_has_model_interface():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(500, 12, 2)), dtype=torch.float32)
    y = (0.7 * x[:, -1, 0] + 0.2 * x[:, -2, 0]).unsqueeze(1)
    pred = RidgeRegression(alpha=1e-3).fit(x, y)(x)
    assert pred.shape == (500, 1) and torch.allclose(pred, y, atol=1e-2)


def test_metrics_are_per_patient_then_averaged():
    pred = np.array([0.0, 0.0, 10.0, 10.0])
    true = np.array([3.0, 3.0, 10.0, 12.0])
    table = per_patient_metrics(pred, true, [1, 1, 2, 2])
    assert table.set_index("patient")["rmse"].to_dict() == pytest.approx({1: 3.0, 2: np.sqrt(2)})
    s = summarize_patients(table)
    assert s["rmse_mean"] == pytest.approx((3 + np.sqrt(2)) / 2)  # mean of patients, not pooled
    assert rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
