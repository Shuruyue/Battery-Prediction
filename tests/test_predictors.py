"""Predictors: closed-form cases and the finite-sample guarantee of split conformal intervals."""
import numpy as np
import pytest

from ups_battery_prediction import predictors as pr
from ups_battery_prediction.design import NORMAL


def test_peukert_passes_through_both_rated_points():
    m = pr.Peukert().fit_two_points(0.5, 120.0, 1.0, 51.0)
    assert np.allclose(m.predict([[25, 0.5, 1.0], [25, 1.0, 1.0]]), [120.0, 51.0], rtol=1e-12)
    assert m.predict([[25, 1.0, 0.8]])[0] == 0.8 * 51.0


def test_physics_features_recover_an_exact_power_law():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(0, 45, 50), rng.uniform(0.2, 1.5, 50), rng.uniform(0.7, 1.0, 50)])
    y = 2.0 * X[:, 1] ** -1.1 * X[:, 2] ** 1.0 * np.exp(-300.0 / (X[:, 0] + 273.15))
    m = pr.PhysicsFeatures().fit(X, y)
    assert np.allclose(m.predict(X), y, rtol=1e-10)
    assert np.allclose(m.reg.coef_, [-1.1, 1.0, -0.3], atol=1e-10)


def test_conformal_quantile_is_the_k_th_smallest_relative_residual():
    # n = 50, alpha = 0.1: k = ceil(51 x 0.9) = 46; residuals 0.01, 0.02, ..., 0.50 in shuffled order
    rng = np.random.default_rng(0)
    r = rng.permutation(np.arange(1, 51) / 100)
    assert pr.conformal_quantile(1.0 + r, np.ones(50), alpha=0.1) == pytest.approx(0.46, abs=1e-15)
    # n = 9, alpha = 0.1: (n + 1)(1 - alpha) is exactly 9, so k = 9 (not 10, which does not exist)
    assert pr.conformal_quantile(1.0 + np.arange(1, 10) / 10, np.ones(9), alpha=0.1) == pytest.approx(0.9, abs=1e-15)
    assert pr.conformal_quantile(np.full(5, 1.1), np.ones(5), alpha=0.1) == np.inf


def test_conformal_coverage_is_k_over_n_plus_one_on_exchangeable_data():
    # for continuous residuals the coverage of a new point is exactly k / (n + 1) = 46/51 = 0.902
    rng = np.random.default_rng(2)
    trials = 40000
    y = 50.0 * (1 + 0.05 * rng.standard_normal((trials, 51)))
    yhat = np.full(51, 50.0)
    covered = [abs(row[50] - 50.0) <= pr.conformal_quantile(row[:50], yhat[:50], alpha=0.1) * 50.0 for row in y]
    se = np.sqrt(0.902 * 0.098 / trials)                                     # 0.0015
    assert abs(np.mean(covered) - 46 / 51) < 4 * se                         # 47/51 = 0.922 is 13 se away


def test_distance_outside_is_zero_inside_and_counts_box_widths_outside():
    inside = [[20.0, 0.5, 0.95], [NORMAL["T"][1], NORMAL["load"][0], 1.0]]
    assert np.all(pr.distance_outside(inside) == 0.0)
    w = NORMAL["T"][1] - NORMAL["T"][0]
    assert pr.distance_outside([[NORMAL["T"][1] + 2 * w, 0.5, 0.95]])[0] == 2.0
