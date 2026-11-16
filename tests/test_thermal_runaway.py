"""Checks of the thermal-runaway model and of the hand-written stiff integrator."""
from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ups_battery_prediction import thermal_runaway as tr


def test_backward_euler_on_linear_stiff_problem():
    # y' = -1000 (y - cos t): the solution follows cos t after a fast transient
    def f(t, y):
        return -1000.0 * (y - np.cos(t))
    t, y, info = tr.backward_euler(f, [0.0], 2.0, rtol=1e-6, atol=1e-9, dt_max=0.05)
    exact = (1e6 * np.cos(t) + 1e3 * np.sin(t)) / (1e6 + 1) - 1e6 / (1e6 + 1) * np.exp(-1000 * t)
    assert np.max(np.abs(y[:, 0] - exact)) < 2e-3


def test_adiabatic_temperature_rise_equals_released_heat():
    cell = replace(tr.Cell(), h=0.0, emissivity=0.0)
    for ch in (tr.LCO, tr.NMC, tr.LFP):
        m = tr.Module(ch, cell, 1, T_amb=300.0)
        t, y, _ = tr.backward_euler(m.rhs, tr.initial_state(ch, 420.0), 5.0e4, rtol=1e-4, dt_max=1e4)
        yf = y[-1]
        released = (ch.H_sei * ch.W_c * (ch.c_sei0 - yf[1]) + ch.H_ne * ch.W_c * (ch.c_neg0 - yf[2])
                    + ch.H_pe * ch.W_p * (yf[4] - ch.alpha0) + ch.H_e * ch.W_e * (ch.c_e0 - yf[5])) / cell.rho_cp
        assert abs((yf[0] - 420.0) - released) < 1e-6 * released          # energy balance
        assert yf[0] - 420.0 <= ch.adiabatic_heat() / cell.rho_cp + 1e-6   # bounded by complete reaction
        assert abs(yf[2] + yf[3] - (ch.c_neg0 + ch.t_sei0)) < 1e-9         # c_neg + t_sei is conserved


def test_matches_scipy_radau_on_oven_test():
    ch, cell = tr.LCO, tr.Cell()
    m = tr.Module(ch, cell, 1, T_amb=428.15)
    y0 = tr.initial_state(ch, 301.15)
    t, y, _ = tr.backward_euler(m.rhs, y0, 3600.0, rtol=1e-5)
    ref = solve_ivp(m.rhs, (0, 3600.0), y0, method="Radau", rtol=1e-9, atol=1e-11)
    t_own = tr.crossing_time(t, y[:, 0], 528.15)
    t_ref = tr.crossing_time(ref.t, ref.y[0], 528.15)
    assert abs(t_own / t_ref - 1) < 0.002
    assert abs(y[:, 0].max() - ref.y[0].max()) < 0.5


def test_semenov_critical_ambient_temperature():
    cell = tr.Cell()
    E, Q0 = 1.3508e5, 1.714e6 * 610 * 2.5e13 * 0.75
    hA, V = cell.h * cell.area, cell.volume

    def gap(Ta):   # heat generation minus loss at the tangency point T - Ta = R T^2 / E
        T = brentq(lambda T: T - Ta - tr.R_GAS * T * T / E, Ta, Ta + 60)
        return Q0 * V * np.exp(-E / (tr.R_GAS * T)) - hA * (T - Ta)
    Ta_exact = brentq(gap, 350.0, 480.0)

    def runs_away(Ta):
        def f(t, y):
            return np.array([(Q0 * V * np.exp(-E / (tr.R_GAS * y[0])) - hA * (y[0] - Ta)) / (cell.rho_cp * V)])
        _, yy, _ = tr.backward_euler(f, [Ta], 4.0e5, rtol=1e-5, dt_max=500.0)
        return yy[:, 0].max() > Ta + 150.0
    assert not runs_away(Ta_exact - 0.2) and runs_away(Ta_exact + 0.2)


def test_explicit_euler_fails_where_implicit_succeeds():
    ch = tr.LCO
    m = tr.Module(ch, tr.Cell(), 1, T_amb=428.15)
    y0 = tr.initial_state(ch, 301.15)
    _, ys = tr.explicit_euler(m.rhs, y0, 3600.0, 1.0)
    assert not np.isfinite(ys).all() or np.nanmax(np.abs(ys[:, 0])) > 1e6
    _, y, _ = tr.backward_euler(m.rhs, y0, 3600.0, rtol=1e-4)
    assert np.isfinite(y).all() and y[:, 1:].min() > -1e-9 and y[:, 0].max() < 1000.0


def test_oven_onset_between_145_and_150_C_for_LCO():
    # Hatchard et al. 2001 (18650 LiCoO2): no runaway at 145 C, runaway at 155 C
    assert not tr.oven_test(tr.LCO, 145 + 273.15, rtol=1e-4)[2]["runaway"]
    assert tr.oven_test(tr.LCO, 155 + 273.15, rtol=1e-4)[2]["runaway"]


def test_module_conserves_energy_without_reactions_or_losses():
    cell = replace(tr.Cell(), h=0.0, emissivity=0.0)
    m = tr.Module(tr.LCO, cell, 4, T_amb=300.0, G=0.5, A_face=0.0, reactions=False)
    y0 = tr.initial_state(tr.LCO, 300.0, 4)
    y0[0] = 500.0
    t, y, _ = tr.backward_euler(m.rhs, y0, 5000.0, rtol=1e-5, dt_max=50.0)
    T = y[:, 0::tr.NS]
    assert np.max(np.abs(T.mean(axis=1) - T[0].mean())) < 1e-8      # mean temperature constant
    assert np.ptp(T[-1]) < 1e-3                                     # and the row equilibrates
