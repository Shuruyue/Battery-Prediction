"""Checks of the thermal-runaway model and of its hand-written stiff integrator.

1. Adiabatic cell: the temperature rise equals the heat released by the reactions (energy balance).
2. Oven test at 155 C (LiCoO2): own adaptive backward Euler at several tolerances against SciPy's Radau.
3. Explicit Euler with fixed steps on the same problem (it overflows at the runaway with a 1 s step).
4. Semenov theory: one zeroth-order reaction, critical ambient temperature from the tangency condition
   against the one found by integrating the ODE.
5. Oven tests of the three chemistries and their critical oven temperature (bisection).

Writes results/verify_runaway.json. About ten minutes on one core.
"""
from dataclasses import replace

import _path  # noqa: F401
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ups_battery_prediction import reference as ref
from ups_battery_prediction import thermal_runaway as tr

CHEMS = (tr.LCO, tr.NMC, tr.LFP)


def adiabatic():
    out = {}
    ad = replace(tr.Cell(), h=0.0, emissivity=0.0)
    for ch in CHEMS:
        m = tr.Module(ch, ad, 1, T_amb=300.0)
        t, y, info = tr.backward_euler(m.rhs, tr.initial_state(ch, 420.0), 2.0e5, rtol=1e-5, dt_max=1e4)
        yf = y[-1]
        released = (ch.H_sei * ch.W_c * (ch.c_sei0 - yf[1]) + ch.H_ne * ch.W_c * (ch.c_neg0 - yf[2])
                    + ch.H_pe * ch.W_p * (yf[4] - ch.alpha0) + ch.H_e * ch.W_e * (ch.c_e0 - yf[5])) / ad.rho_cp
        out[ch.name] = dict(dT_complete=ch.adiabatic_heat() / ad.rho_cp, dT_sim=float(yf[0] - 420.0),
                            dT_from_composition=float(released), steps=info["steps"])
        print("adiabatic", ch.name, out[ch.name], flush=True)
    return out


def oven155():
    ch, cell = tr.LCO, tr.Cell()
    m = tr.Module(ch, cell, 1, T_amb=428.15)
    y0 = tr.initial_state(ch, 301.15)
    res = {}
    for tol in (1e-3, 1e-4, 1e-5, 1e-6):
        tt, yy, inf = tr.backward_euler(m.rhs, y0, 7200.0, rtol=tol)
        res[f"{tol:g}"] = dict(t_runaway=tr.crossing_time(tt, yy[:, 0], 528.15), T_peak=float(yy[:, 0].max()),
                               steps=inf["steps"])
    rd = solve_ivp(m.rhs, (0, 7200.0), y0, method="Radau", rtol=1e-10, atol=1e-12)
    res["radau"] = dict(t_runaway=tr.crossing_time(rd.t, rd.y[0], 528.15), T_peak=float(rd.y[0].max()),
                        steps=int(rd.t.size))
    print("oven 155 C", res, flush=True)
    ee = {}
    for dt in (1.0, 0.1, 0.01):
        ts, ys = tr.explicit_euler(m.rhs, y0, 3600.0, dt)
        fin = np.isfinite(ys).all(axis=1) & (np.abs(ys[:, 0]) < 1e6)
        ee[f"{dt:g}"] = dict(stays_physical=bool(fin.all()),
                             first_bad_time=float(ts[np.argmax(~fin)]) if not fin.all() else None)
    print("explicit Euler", ee, flush=True)
    return res, ee


def semenov():
    cell = tr.Cell()
    E, Q0 = tr.LCO.E_ne, tr.LCO.H_ne * tr.LCO.W_c * tr.LCO.A_ne * tr.LCO.c_neg0   # W/m3 prefactor, no depletion
    hA, V = cell.h * cell.area, cell.volume

    def gap(Ta):     # heat generation minus loss at the tangency point T - Ta = R T^2 / E
        T = brentq(lambda T: T - Ta - tr.R_GAS * T * T / E, Ta, Ta + 60)
        return Q0 * V * np.exp(-E / (tr.R_GAS * T)) - hA * (T - Ta)

    def runs_away(Ta):
        def f(t, y):
            return np.array([(Q0 * V * np.exp(-E / (tr.R_GAS * y[0])) - hA * (y[0] - Ta)) / (cell.rho_cp * V)])
        _, yy, _ = tr.backward_euler(f, [Ta], 4.0e5, rtol=1e-6, dt_max=500.0)
        return yy[:, 0].max() > Ta + 150.0

    Ta = brentq(gap, 350.0, 480.0)
    lo, hi = Ta - 5, Ta + 5
    while hi - lo > 0.01:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if runs_away(mid) else (mid, hi)
    out = dict(Ta_crit_tangency=Ta, Ta_crit_ode=0.5 * (lo + hi))
    print("Semenov", out, flush=True)
    return out


def ovens():
    ov = {}
    for ch in CHEMS:
        for Tc in (140, 145, 150, 155, 160, 170, 180, 200):
            _, _, s = tr.oven_test(ch, Tc + 273.15, t_end=10800.0)
            ov[f"{ch.name}_{Tc}"] = dict(runaway=s["runaway"], T_peak_C=s["T_peak"] - 273.15,
                                        t_runaway_min=s["t_runaway_s"] / 60)
        ov[f"{ch.name}_critical_C"] = tr.critical_oven_temperature(ch) - 273.15
        print(ch.name, "critical oven temperature", round(ov[f"{ch.name}_critical_C"], 2), "C", flush=True)
    return ov


def main():
    o155, ee = oven155()
    out = dict(adiabatic=adiabatic(), oven155_solver=o155, explicit_euler=ee, semenov=semenov(), oven=ovens())
    ref.save_json(out, _path.result("verify_runaway.json"), indent=1)


if __name__ == "__main__":
    main()
