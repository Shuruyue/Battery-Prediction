"""Reduced-order discharge model of the LG M50 cell, written from scratch: SPM or SPMe, isothermal or lumped.

One spherical particle per electrode (single-particle model, SPM). With ``model="SPMe"`` the salt
concentration across the cell is solved as well and the terminal voltage gains the electrolyte and solid ohmic
terms of the single-particle model with electrolyte (SPMe; S. G. Marquis et al., J. Electrochem. Soc. 166 (2019)
A3693, doi:10.1149/2.0341915jes):

    V = U_p(y_s, T) - U_n(x_s, T) - eta_n - eta_p + dphi_e + dphi_s - I R_film
    eta_k  = mean over electrode k of (2RT/F) asinh( i_k / (2 j0_k(c_s,surf, c_e(x), T)) ),  i_k = I/(a_k L_k A)
    dphi_e = mean_p phi_e - mean_n phi_e,
             d phi_e/dx = chi(c_e, T) (RT/F) d ln c_e/dx - i_e / (eps^1.5 kappa_e(c_e, T))
    dphi_s = -i ( L_n/(3 sigma_n) + L_p/(3 sigma_p) )

The electrolyte potential is integrated with chi and kappa_e evaluated at the local concentration, as in the
'integrated' form of F. Brosa Planella, M. Sheikh, W. D. Widanage, Electrochim. Acta 388 (2021) 138524,
doi:10.1016/j.electacta.2021.138524. PyBaMM's SPMe evaluates both at its x-averaged concentration instead, and its
discretisation returns 1036 mol/m3 for that average when the concentration is 1000 mol/m3 everywhere, so the
code check against PyBaMM (scripts/verify_pybamm.py) uses a parameter variant in which chi and kappa_e do not
depend on concentration; there the two forms are the same equations.

U(x, T) = U_ref(x) + (T - T_ref) dU/dT(x). ``model="SPM"`` drops dphi_e and dphi_s and evaluates j0 at
the resting electrolyte concentration (PyBaMM's SPM).

With ``thermal="lumped"`` the cell is one thermal node (PyBaMM's lumped model):

    C(T) dT/dt = I (U_p - U_n - V) + I T (dU_n/dT - dU_p/dT) - h A_cool (T - T_amb),

the first term being all irreversible heat (reaction, ohmic, electrolyte polarisation, SEI film) and the
second the reversible (entropic) heat. ``thermal="isothermal"`` holds T at the ambient value. h may differ
between the points of a batch.

Time stepping is backward Euler for particles and electrolyte with coefficients from the previous step; the
temperature follows explicitly from the heat of the step. The load is a constant current or a constant
power (the load of an inverter); for constant power the current is found by Newton iteration on I V(I) = P.
"""
from dataclasses import dataclass

import numpy as np

from . import params as p
from .ageing import PARAMETERS_USED, V_FULL, cell_state
from .numerics import Electrolyte, Particle, step_pair

_EPS = 1e-6
_PARTICLES = {}


@dataclass(frozen=True)
class Scale:
    """Factors on parameter groups, for sensitivity studies: solid diffusivities, exchange currents and the
    ohmic resistances (electrolyte and solid conductivities are divided by `ohmic`)."""
    diffusivity: float = 1.0
    exchange: float = 1.0
    ohmic: float = 1.0


def _particles(prm, n_r):
    """The two particle grids (cached: building the Kirchhoff table of D_ref takes a moment)."""
    key = (n_r, prm.R_N, prm.R_P, prm.D_n_ref, prm.D_p_ref)
    if key not in _PARTICLES:
        _PARTICLES[key] = (Particle(prm.R_N, n_r, prm.D_n_ref), Particle(prm.R_P, n_r, prm.D_p_ref))
    return _PARTICLES[key]


def _asinh_eta(i, j0, T, prm):
    return (2.0 * prm.R * T / prm.F) * np.arcsinh(i / (2.0 * j0))


def _constant_power_current(voltage, I, load, alive, s):
    """Newton iteration on I V(I) = P for every point of the batch, from the current of the previous step.
    Returns I, V(I), the surface stoichiometries and whether the power is met. Once the electrolyte of a cell is
    exhausted its voltage is -inf; such a cell fails the check and its discharge ends."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(30):
            V, xs, ys = voltage(I, s)
            f = I * V - load
            if np.all(np.abs(f[alive]) < 1e-10 * np.maximum(load[alive], 1.0)):
                break
            hI = 1e-6 * np.maximum(I, 1e-3)
            V2, _, _ = voltage(I + hI, s)
            df = ((I + hI) * V2 - I * V) / hI
            stepI = np.where(df > 1e-9, f / np.where(df > 1e-9, df, 1.0), 0.0)
            I = np.clip(I - stepI, 0.2 * I, 5.0 * I)
        ok = np.isfinite(V) & (np.abs(I * V - load) < 1e-6 * np.maximum(load, 1.0))
    return I, V, xs, ys, ok


def simulate(T_amb, soh=1.0, power=None, current=None, model="SPMe", thermal="lumped", dt=1.0, n_r=30,
             n_x=(20, 10, 20), h=p.H_COOL, t_max=None, v_min=p.V_MIN, scale=Scale(), film=True,
             return_traces=False, prm=p):
    """Discharge from full charge at constant power [W] or constant current [A] per cell.

    T_amb [K] (the cell starts at the ambient temperature), soh and the load broadcast to a common batch.
    Returns a dict with 'runtime_s' (time until V = v_min, or until the power can no longer be delivered),
    'energy_Wh', 'charge_Ah', 'T_end_K', 'T_max_K' and, optionally, the traces. `prm` is the parameter set
    (the params module, or a variant of it from params.variant); the state of an aged cell always comes from the
    params module, so a variant may not change what ageing.cell_state uses."""
    if model not in ("SPM", "SPMe") or thermal not in ("isothermal", "lumped"):
        raise ValueError("model must be 'SPM' or 'SPMe', thermal 'isothermal' or 'lumped'")
    if (power is None) == (current is None):
        raise ValueError("give exactly one of power and current")
    clash = getattr(prm, "overrides", frozenset()) & PARAMETERS_USED
    if clash:
        raise ValueError(f"the cell state of ageing.py does not see these overrides: {sorted(clash)}")
    T_amb = np.atleast_1d(np.asarray(T_amb, float))
    soh = np.atleast_1d(np.asarray(soh, float))
    load = np.atleast_1d(np.asarray(power if power is not None else current, float))
    h = np.atleast_1d(np.asarray(h, float))
    T_amb, soh, load, h = (a.copy() for a in np.broadcast_arrays(T_amb, soh, load, h))
    B = T_amb.size
    states = [cell_state(s) for s in soh]
    x0 = np.array([s.x0 for s in states])
    y0 = np.array([s.y0 for s in states])
    R_c = np.array([s.r_contact for s in states]) * (1.0 if film else 0.0)
    pn, pp = _particles(prm, n_r)
    spme = model == "SPMe"
    el = Electrolyte(prm, *n_x) if spme else None
    lumped = thermal == "lumped"

    T = T_amb.copy()
    th_n = np.repeat(x0[:, None], n_r, 1)
    th_p = np.repeat(y0[:, None], n_r, 1)
    c_e = np.full((B, el.N), prm.C_E0) if spme else None
    kn = 1.0 / (prm.F * prm.A_N * prm.L_N * prm.AREA * prm.C_N_MAX)    # q_n = +I kn
    kp = 1.0 / (prm.F * prm.A_P * prm.L_P * prm.AREA * prm.C_P_MAX)    # q_p = -I kp
    sn = 1.0 / (prm.A_N * prm.L_N * prm.AREA)                          # i_n = I sn
    sp = 1.0 / (prm.A_P * prm.L_P * prm.AREA)
    const_power = power is not None
    if t_max is None:
        ref = load / 3.6 if const_power else load
        t_max = float(np.max(3.0 * 3600.0 * prm.Q_NOMINAL / np.maximum(ref, 1e-9)))

    def voltage(I, s):
        qn, qp = I * kn, -I * kp
        xs = np.clip(pn.surface(s["un"] + qn * s["wn"], qn, s["an"]), _EPS, 1 - _EPS)
        ys = np.clip(pp.surface(s["up"] + qp * s["wp"], qp, s["ap"]), _EPS, 1 - _EPS)
        Tk = s["T"]
        if spme:
            i = I / prm.AREA
            ce = s["ue"] + i[:, None] * s["we"]
            j0n = scale.exchange * prm.j0_n(xs[:, None], Tk[:, None], np.maximum(ce[:, el.in_n], 0.0))
            j0p = scale.exchange * prm.j0_p(ys[:, None], Tk[:, None], np.maximum(ce[:, el.in_p], 0.0))
            eta_n = _asinh_eta((I * sn)[:, None], j0n, Tk[:, None], prm).mean(axis=1)
            eta_p = _asinh_eta((I * sp)[:, None], j0p, Tk[:, None], prm).mean(axis=1)
            dphi_e = el.potential_difference(ce, Tk, i * scale.ohmic)   # scales the resistive part only
            dphi_s = -i * scale.ohmic * (prm.L_N / (3 * prm.SIGMA_N) + prm.L_P / (3 * prm.sigma_p(Tk)))
            extra = dphi_e + dphi_s
        else:
            eta_n = _asinh_eta(I * sn, scale.exchange * prm.j0_n(xs, Tk), Tk, prm)
            eta_p = _asinh_eta(I * sp, scale.exchange * prm.j0_p(ys, Tk), Tk, prm)
            extra = 0.0
        V = prm.ocp_p(ys, Tk) - prm.ocp_n(xs, Tk) - eta_n - eta_p + extra - I * R_c
        return V, xs, ys

    I = load / V_FULL if const_power else load.copy()
    alive = np.ones(B, bool)
    runtime = np.full(B, np.nan)
    energy = np.zeros(B)
    charge = np.zeros(B)
    T_max = T.copy()
    V_prev = np.full(B, V_FULL)
    names = ("t", "V", "I", "T", "xs", "ys")
    tr = {k: [] for k in names} if return_traces else None
    if return_traces:
        for k, v in zip(names, (0.0, V_prev.copy(), np.zeros(B), T.copy(), x0.copy(), y0.copy()), strict=True):
            tr[k].append(v)
    t = 0.0
    while alive.any() and t < t_max:
        an = scale.diffusivity * np.exp(prm.D_N_THETA * (1.0 / prm.T_REF - 1.0 / T))
        ap = scale.diffusivity * np.exp(prm.D_P_THETA * (1.0 / prm.T_REF - 1.0 / T))
        un, wn, up, wp = step_pair(pn, th_n, an[:, None], pp, th_p, ap[:, None], dt)
        s = dict(un=un[:, -1], wn=wn[:, -1], up=up[:, -1], wp=wp[:, -1], an=an, ap=ap, T=T)
        if spme:
            ue, we = el.step(c_e, T, dt)
            s.update(ue=ue, we=we)
        if const_power:
            I, V, xs, ys, ok = _constant_power_current(voltage, I, load, alive, s)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                V, xs, ys = voltage(I, s)
            ok = np.isfinite(V)
        hit = alive & ((V <= v_min) | ~ok)
        if hit.any():
            frac = np.where(ok & (V_prev > V), (V_prev - v_min) / np.where(V_prev > V, V_prev - V, 1.0), 0.0)
            frac = np.clip(frac, 0.0, 1.0)
            runtime[hit] = t + frac[hit] * dt
            energy[hit] += (frac * dt * I * 0.5 * (V_prev + np.maximum(V, v_min)))[hit]
            charge[hit] += (frac * dt * I)[hit]
            alive &= ~hit
        live = alive.copy()
        energy[live] += (dt * I * V)[live]
        charge[live] += (dt * I)[live]
        th_n = un + (I * kn)[:, None] * wn
        th_p = up - (I * kp)[:, None] * wp
        if spme:
            c_e = ue + (I / prm.AREA)[:, None] * we
        if lumped:
            Q = I * (prm.ocp_p(ys, T) - prm.ocp_n(xs, T) - V) + I * T * (prm.dUdT_n(xs) - prm.dUdT_p(ys))
            hA_C = h * prm.COOLING_AREA / prm.heat_capacity(T)
            T_new = (T + dt * (Q / prm.heat_capacity(T) + hA_C * T_amb)) / (1.0 + dt * hA_C)   # cooling implicit
            T = np.where(live, T_new, T)
            T_max = np.maximum(T_max, T)
        V_prev = np.where(live, V, V_prev)
        t += dt
        if return_traces:
            shown = live | hit
            for k, v in zip(names, (t, np.where(shown, V, np.nan), np.where(shown, I, np.nan), T.copy(),
                                    xs.copy(), ys.copy()), strict=True):
                tr[k].append(v)
        I = np.where(alive, I, 0.0)          # a finished cell rests (its states stay physical)
    out = {"runtime_s": runtime, "energy_Wh": energy / 3600.0, "charge_Ah": charge / 3600.0, "T_end_K": T,
           "T_max_K": T_max, "x_mean_end": pn.mean(th_n), "y_mean_end": pp.mean(th_p)}
    if spme:
        out["salt_end"] = el.salt(c_e)
        out["salt_start"] = el.salt(np.full((1, el.N), prm.C_E0))[0]
    if return_traces:
        out.update({k: np.array(v) for k, v in tr.items()})
    return out
