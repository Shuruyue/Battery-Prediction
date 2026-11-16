"""Predictors: closed-form cases and the finite-sample guarantee of split conformal intervals."""
import numpy as np

from ups_battery_prediction import predictors as pr
from ups_battery_prediction.reference import NORMAL


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


def test_conformal_interval_covers_at_least_the_nominal_fraction_on_exchangeable_data():
    rng = np.random.default_rng(2)
    covered = []
    for _ in range(2000):
        yhat = np.full(51, 50.0)
        y = yhat * (1 + 0.05 * rng.standard_normal(51))
        q = pr.conformal_quantile(y[:50], yhat[:50], alpha=0.1)
        covered.append(abs(y[50] - yhat[50]) <= q * yhat[50])
    # guarantee: P(cover) >= 1 - alpha; with n = 50 the expected coverage is 46/51 = 0.902
    assert np.mean(covered) > 0.88


def test_distance_outside_is_zero_inside_and_counts_box_widths_outside():
    inside = [[20.0, 0.5, 0.95], [NORMAL["T"][1], NORMAL["load"][0], 1.0]]
    assert np.all(pr.distance_outside(inside) == 0.0)
    w = NORMAL["T"][1] - NORMAL["T"][0]
    assert pr.distance_outside([[NORMAL["T"][1] + 2 * w, 0.5, 0.95]])[0] == 2.0
