"""Predictors of UPS runtime from (ambient temperature [C], load [fraction of 18 W], state of health).

Three kinds of tool, scored against the same reference:
  1. a formula      - Peukert's law, calibrated on two rated points (25 C, new cell), as from a datasheet
  2. a simulation   - the hand-written reduced-order model (discharge.simulate), no calibration
  3. machine learning trained on noisy 'measurements' from the normal operating range
plus two ways of putting physics into the learning step (physics features, simulation + data correction).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .reference import NORMAL


def distance_outside(X):
    """How far a point lies outside the normal box, in box widths (Euclidean over the three axes)."""
    X = np.atleast_2d(X)
    d = np.zeros(len(X))
    for j, name in enumerate(("T", "load", "soh")):
        lo, hi = NORMAL[name]
        w = hi - lo
        d += (np.clip(lo - X[:, j], 0, None) / w) ** 2 + (np.clip(X[:, j] - hi, 0, None) / w) ** 2
    return np.sqrt(d)


class Peukert:
    """runtime = t_rated SOH load^(-k); t_rated and k from two rated points (new cell, 25 C)."""

    def fit_two_points(self, load_a, t_a, load_b, t_b):
        self.k = float(np.log(t_a / t_b) / np.log(load_b / load_a))
        self.t_rated = float(t_b * load_b ** self.k)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.t_rated * X[:, 2] * X[:, 1] ** (-self.k)


def make_models(seed=0):
    """The machine-learning regressors, untrained."""
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial (degree 3)": make_pipeline(StandardScaler(), PolynomialFeatures(3),
                                               RidgeCV(alphas=np.logspace(-6, 2, 30))),
        "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=seed),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=5,
                                                           random_state=seed),
        "Neural network": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=3000, tol=1e-6, n_iter_no_change=50,
            random_state=seed)),
        "Gaussian process": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            ConstantKernel(1.0, (1e-2, 1e4)) * RBF([1.0, 1.0, 1.0], (1e-1, 1e2)) + WhiteKernel(1e-3, (1e-8, 1e0)),
            normalize_y=True, n_restarts_optimizer=1, random_state=seed)),
    }


class PhysicsFeatures:
    """Law suggested by the physics, fitted by least squares:
        log t = a + b log(load) + c log(SOH) + d / T
    (Peukert exponent b, capacity scaling c, Arrhenius-type temperature term d)."""

    @staticmethod
    def features(X):
        X = np.atleast_2d(X)
        return np.column_stack([np.log(X[:, 1]), np.log(X[:, 2]), 1000.0 / (X[:, 0] + 273.15)])

    def fit(self, X, y):
        self.reg = LinearRegression().fit(self.features(X), np.log(y))
        return self

    def predict(self, X):
        return np.exp(self.reg.predict(self.features(X)))


class Hybrid:
    """Simulation corrected by data: a small linear model learns log(measured / simulated)."""

    def __init__(self, sim_fun, model=None):
        self.sim_fun = sim_fun
        self.model = model or make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))

    def fit(self, X, y):
        self.model.fit(X, np.log(y / self.sim_fun(X)))
        return self

    def predict(self, X):
        return self.sim_fun(X) * np.exp(self.model.predict(X))


def conformal_quantile(y_cal, yhat_cal, alpha=0.1):
    """Split-conformal half-width factor q for relative residuals: the interval is yhat (1 +- q), q being the
    ceil((n + 1)(1 - alpha))-th smallest of |y - yhat| / yhat on calibration data."""
    r = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)) / np.asarray(yhat_cal)
    n = len(r)
    return float(np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def mape(y, yhat):
    return float(100.0 * np.mean(np.abs(np.asarray(yhat) - y) / y))
