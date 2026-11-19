"""Finite-volume building blocks against closed-form solutions and conservation laws."""
import numpy as np
from scipy.optimize import brentq

from battery_prediction import params as p
from battery_prediction.numerics import Electrolyte, Particle, thomas


def test_thomas_matches_dense_solver():
    rng = np.random.default_rng(0)
    n = 12
    a, c = rng.uniform(-1, 0, (3, n)), rng.uniform(-1, 0, (3, n))
    b = 2.5 + rng.uniform(0, 1, (3, n))
    d = rng.normal(size=(3, n))
    x = thomas(a, b, c, d)
    for k in range(3):
        M = np.diag(b[k]) + np.diag(a[k, 1:], -1) + np.diag(c[k, :-1], 1)
        assert np.allclose(x[k], np.linalg.solve(M, d[k]), rtol=1e-12, atol=1e-13)


# --- particle -------------------------------------------------------------------------------------------------
def _sphere_surface_exact(tau, n_terms=200):
    """Constant surface flux out of a sphere, constant D (Crank, The Mathematics of Diffusion, eq. 6.45):
    (theta_s - theta_0) D / (q R) = -(3 tau + 1/5 - 2 sum exp(-lam^2 tau) / lam^2), tan(lam) = lam."""
    lam = np.array([brentq(lambda x: np.tan(x) - x, (k + 0.5) * np.pi - 1.5, (k + 0.5) * np.pi - 1e-9)
                    for k in range(1, n_terms + 1)])
    return -(3 * tau + 0.2 - 2 * np.sum(np.exp(-lam ** 2 * tau) / lam ** 2))


D0, RAD, Q0, TH0, T_END = 1e-14, 5e-6, 2e-10, 0.8, 600.0


def _run_constant_D(ns, dt):
    """Surface and mean stoichiometry after T_END for several grids at once."""
    out = []
    for n in ns:
        part = Particle(RAD, n, lambda th: D0 * np.ones_like(np.asarray(th, float)), m=2001)
        th = np.full((1, n), TH0)
        arr = np.ones((1, 1))
        for _ in range(int(round(T_END / dt))):
            u, w = part.step(th, arr, dt)
            th = u + Q0 * w
        out.append((float(part.surface(th[:, -1], Q0, arr[:, 0])[0]), float(part.mean(th)[0])))
    return out


def test_sphere_diffusion_is_second_order_in_space_against_the_series_solution():
    exact = TH0 + Q0 * RAD / D0 * _sphere_surface_exact(D0 * T_END / RAD ** 2)
    errs = [abs(s - exact) for s, _ in _run_constant_D((10, 20, 40), 0.04)]
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert errs[2] < 6e-5 and min(order) > 1.8, (errs, order)


def test_lithium_is_conserved_to_round_off():
    _, mean = _run_constant_D((20,), 2.0)[0]
    assert abs(mean - (TH0 - 3 * Q0 * T_END / RAD)) < 1e-13


# --- electrolyte ----------------------------------------------------------------------------------------------
D_CONST, T_PLUS = 3e-10, 0.3
CONST = p.variant(D_e=lambda c, T: D_CONST + 0.0 * np.asarray(c, float),
                  t_plus=lambda c, T: T_PLUS + 0.0 * np.asarray(c, float))


def _steady_exact(x, i):
    """Steady salt profile for constant D_e and t+: -eps^b D dc/dx = (1 - t+) i_e(x) / F, mean of eps c fixed."""
    k = (1 - T_PLUS) * i / (p.F * D_CONST)
    tn, ts, tp = p.EPS_E_N ** p.BRUG, p.EPS_E_S ** p.BRUG, p.EPS_E_P ** p.BRUG
    x1 = p.L_N + p.L_S
    c_ln = -k * p.L_N / (2 * tn)
    c_x1 = c_ln - k * p.L_S / ts

    def shape(z):
        return np.where(z <= p.L_N, -k * z ** 2 / (2 * p.L_N * tn),
                        np.where(z <= x1, c_ln - k * (z - p.L_N) / ts,
                                 c_x1 - k / (p.L_P * tp) * (p.L_X * (z - x1) - (z ** 2 - x1 ** 2) / 2)))
    zz = np.linspace(0, p.L_X, 400001)
    eps = np.where(zz <= p.L_N, p.EPS_E_N, np.where(zz <= x1, p.EPS_E_S, p.EPS_E_P))
    total_eps = p.EPS_E_N * p.L_N + p.EPS_E_S * p.L_S + p.EPS_E_P * p.L_P
    c0 = p.C_E0 - np.trapezoid(eps * shape(zz), zz) / total_eps
    return c0 + shape(x)


def _steady(el, i, T=298.15):
    c = np.full((1, el.N), p.C_E0)
    for _ in range(60):
        u, w = el.step(c, np.array([T]), 1e5)
        c = u + i * w
    return c


def test_electrolyte_steady_state_against_the_closed_form():
    i = 30.0  # A/m2
    errs = []
    for n in (10, 20, 40):
        el = Electrolyte(CONST, n, n // 2, n)
        c = _steady(el, i)
        errs.append(np.max(np.abs(c[0] - _steady_exact(el.x, i))))
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert errs[2] < 0.5 and min(order) > 1.8, (errs, order)     # mol/m3, on a profile spanning ~700 mol/m3


def test_electrolyte_conserves_salt_with_the_real_properties():
    el = Electrolyte(p, 20, 10, 20)
    c = np.full((2, el.N), p.C_E0)
    T = np.array([273.15, 318.15])
    start = el.salt(c)
    for k in range(300):
        u, w = el.step(c, T, 2.0)
        c = u + (40.0 if k < 150 else -15.0) * w
    assert np.all(np.abs(el.salt(c) - start) < 1e-12 * start)
    assert c.min() > 0 and np.ptp(c, axis=1).min() > 100.0      # a real gradient was built up


def test_potential_difference_reduces_to_the_composite_form_for_constant_properties():
    T, i = 298.15, 30.0
    chi0, kap0 = 3.0, 0.9
    q = p.variant(chi=lambda c, T: chi0 + 0.0 * np.asarray(c, float), kappa_e=lambda c, T: kap0 + 0.0 * np.asarray(c,
                                                                                                                   float),
                  D_e=CONST.D_e, t_plus=CONST.t_plus)
    errs = []
    for n in (10, 20, 40):
        el = Electrolyte(q, n, n // 2, n)
        c = _steady(el, i)
        got = el.potential_difference(c, np.array([T]), np.array([i]))[0]
        lnc = np.log(c[0])
        Mn = lnc[el.in_n] @ el.h[el.in_n] / p.L_N
        Mp = lnc[el.in_p] @ el.h[el.in_p] / p.L_P
        tn, ts, tp = (e ** p.BRUG for e in (p.EPS_E_N, p.EPS_E_S, p.EPS_E_P))
        composite = chi0 * p.R * T / p.F * (Mp - Mn) - i * (p.L_N / (3 * kap0 * tn) + p.L_S / (kap0 * ts)
                                                           + p.L_P / (3 * kap0 * tp))
        errs.append(abs(got - composite))
    assert errs[2] < 2e-5 and errs[0] / errs[2] > 12, errs    # volts; second order
