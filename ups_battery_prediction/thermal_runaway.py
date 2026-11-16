"""Lumped thermal-abuse model of a lithium-ion cell and of a row of cells.

Four exothermic reactions (Hatchard et al. 2001; Kim, Pesaran and Spotnitz 2007):

    SEI decomposition        dc_sei/dt = -A_sei exp(-E_sei/RT) c_sei
    negative / electrolyte   dc_neg/dt = -A_ne exp(-t_sei/t_ref) exp(-E_ne/RT) c_neg,  dt_sei/dt = -dc_neg/dt
    positive / electrolyte   d(alpha)/dt = A_pe alpha (1 - alpha) exp(-E_pe/RT)
    electrolyte              dc_e/dt = -A_e exp(-E_e/RT) c_e

    rho cp V dT/dt = V (H_sei W_c R_sei + H_ne W_c R_ne + H_pe W_p R_pe + H_e W_e R_e)
                     - h A (T - T_amb) - eps sigma A (T^4 - T_amb^4) + P_ext + exchange with neighbours

Sources
  T. D. Hatchard, D. D. MacNeil, A. Basu, J. R. Dahn, J. Electrochem. Soc. 148 (2001) A755,
    doi:10.1149/1.1377592
  G.-H. Kim, A. Pesaran, R. Spotnitz, J. Power Sources 170 (2007) 476,
    doi:10.1016/j.jpowsour.2007.04.018   (LiCoO2/graphite parameters, 18650 properties)
  P. Peng, F. Jiang, Int. J. Heat Mass Transfer 103 (2016) 1008,
    doi:10.1016/j.ijheatmasstransfer.2016.07.088   (positive-electrode parameters of other cathodes)

The time integrator (backward Euler, Newton iteration, step-size control) is written here from scratch;
scripts/verify_runaway.py compares it with SciPy's Radau method.

The cell of this part is an 18650 LiCoO2/graphite cell, not the LG M50 of the discharge model: Kim et al. give a
complete, published set of abuse kinetics for it, and Hatchard et al. measured its oven tests. The NMC111 and
LiFePO4 variants change only the positive-electrode reaction (Peng and Jiang 2016) and keep the graphite,
electrolyte and cell of Kim et al. No NMC811 kinetics of the same form were found, so the high-nickel cathode of
the discharge model is not represented here.
"""
from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314462618
SIGMA_SB = 5.670374419e-8


@dataclass(frozen=True)
class Chemistry:
    name: str
    A_pe: float      # 1/s
    E_pe: float      # J/mol
    H_pe: float      # J/kg
    W_p: float       # kg/m3
    # shared (graphite negative, carbonate electrolyte), Kim et al. 2007
    A_sei: float = 1.667e15
    E_sei: float = 1.3508e5
    H_sei: float = 2.57e5
    A_ne: float = 2.5e13
    E_ne: float = 1.3508e5
    H_ne: float = 1.714e6
    A_e: float = 5.14e25
    E_e: float = 2.74e5
    H_e: float = 1.55e5
    W_c: float = 6.10e2
    W_e: float = 4.06e2
    c_sei0: float = 0.15
    c_neg0: float = 0.75
    t_sei0: float = 0.033
    t_ref: float = 0.033
    alpha0: float = 0.04
    c_e0: float = 1.0

    def adiabatic_heat(self):
        """Heat released per cell volume if every reaction runs to completion, J/m3."""
        return (self.H_sei * self.W_c * self.c_sei0 + self.H_ne * self.W_c * self.c_neg0
                + self.H_pe * self.W_p * (1.0 - self.alpha0) + self.H_e * self.W_e * self.c_e0)


LCO = Chemistry("LCO", A_pe=6.667e13, E_pe=1.396e5, H_pe=3.14e5, W_p=1.3e3)
NMC = Chemistry("NMC111", A_pe=2.25e14, E_pe=1.54e5, H_pe=7.9e5, W_p=1.293e3)
LFP = Chemistry("LFP", A_pe=2.0e8, E_pe=1.03e5, H_pe=1.947e5, W_p=0.96e3)
CHEMISTRIES = {c.name: c for c in (LCO, NMC, LFP)}


@dataclass(frozen=True)
class Cell:
    """18650 cell (Kim et al. 2007). h = 7.17 W/(m2 K) and a start at 28 C are the oven-test conditions of
    Hatchard et al. 2001 as used by Lopez et al., J. Electrochem. Soc. 162 (2015) A2163,
    doi:10.1149/2.0751510jes; the emissivity 0.8 is that of Hatchard et al. as cited by Chen et al.,
    J. Electrochem. Soc. 168 (2021) 010502, doi:10.1149/1945-7111/abd64c."""
    radius: float = 0.009
    height: float = 0.065
    rho_cp: float = 2.789e6      # J/(m3 K)
    h: float = 7.17              # W/(m2 K)
    emissivity: float = 0.8      # Hatchard et al. 2001, as cited by Chen et al. 2021

    @property
    def volume(self):
        return np.pi * self.radius ** 2 * self.height

    @property
    def area(self):
        return 2 * np.pi * self.radius * self.height + 2 * np.pi * self.radius ** 2


NS = 6  # T, c_sei, c_neg, t_sei, alpha, c_e


def initial_state(chem, T0, n=1):
    y = np.empty((n, NS))
    y[:, 0] = T0
    y[:, 1:] = (chem.c_sei0, chem.c_neg0, chem.t_sei0, chem.alpha0, chem.c_e0)
    return y.ravel()


def rates(chem, y):
    """Reaction rates (1/s) and volumetric heat release (W/m3); y has shape (n, 6)."""
    T = np.maximum(y[:, 0], 1.0)
    r_sei = chem.A_sei * np.exp(-chem.E_sei / (R_GAS * T)) * y[:, 1]
    r_ne = chem.A_ne * np.exp(-y[:, 3] / chem.t_ref) * np.exp(-chem.E_ne / (R_GAS * T)) * y[:, 2]
    r_pe = chem.A_pe * y[:, 4] * (1.0 - y[:, 4]) * np.exp(-chem.E_pe / (R_GAS * T))
    r_e = chem.A_e * np.exp(-chem.E_e / (R_GAS * T)) * y[:, 5]
    q = (chem.H_sei * chem.W_c * r_sei + chem.H_ne * chem.W_c * r_ne
         + chem.H_pe * chem.W_p * r_pe + chem.H_e * chem.W_e * r_e)
    return r_sei, r_ne, r_pe, r_e, q


@dataclass
class Module:
    """A row of n identical cells. Neighbours exchange heat through a conductance G [W/K]
    and by radiation between facing areas A_face [m2]."""
    chem: Chemistry
    cell: Cell = field(default_factory=Cell)
    n: int = 1
    T_amb: float = 298.15
    G: float = 0.0
    A_face: float = 0.0
    p_ext: object = None          # function t -> array of external heating power per cell, W
    reactions: bool = True

    def rhs(self, t, yflat):
        y = yflat.reshape(self.n, NS)
        c, ch = self.cell, self.chem
        T = y[:, 0]
        dy = np.zeros_like(y)
        q = 0.0
        if self.reactions:
            r_sei, r_ne, r_pe, r_e, q = rates(ch, y)
            dy[:, 1] = -r_sei
            dy[:, 2] = -r_ne
            dy[:, 3] = r_ne
            dy[:, 4] = r_pe
            dy[:, 5] = -r_e
        loss = c.h * c.area * (T - self.T_amb) + c.emissivity * SIGMA_SB * c.area * (T ** 4 - self.T_amb ** 4)
        power = q * c.volume - loss
        if self.p_ext is not None:
            power = power + self.p_ext(t)
        if self.n > 1 and (self.G > 0 or self.A_face > 0):
            ex = self.G * (T[1:] - T[:-1]) + c.emissivity * SIGMA_SB * self.A_face * (T[1:] ** 4 - T[:-1] ** 4)
            power[:-1] += ex
            power[1:] -= ex
        dy[:, 0] = power / (c.rho_cp * c.volume)
        return dy.ravel()


# --- time integrators ------------------------------------------------------------------
def _jacobian(f, t, y, f0):
    n = y.size
    J = np.empty((n, n))
    for k in range(n):
        h = 1e-7 * max(abs(y[k]), 1e-3)
        yk = y.copy()
        yk[k] += h
        J[:, k] = (f(t, yk) - f0) / h
    return J


def backward_euler(f, y0, t_end, rtol=1e-4, atol=1e-8, dt0=1e-3, dt_max=10.0, max_steps=2_000_000):
    """Adaptive backward Euler. Each step solves y1 = y0 + dt f(t1, y1) by Newton iteration;
    the step size follows the difference between an explicit predictor and the implicit
    result, which estimates the local error of the first-order method."""
    t, y = 0.0, np.array(y0, float)
    ts, ys = [t], [y.copy()]
    dt = dt0
    fy = f(t, y)
    nsteps = nrej = 0
    while t < t_end and nsteps < max_steps:
        dt = min(dt, t_end - t, dt_max)
        t1 = t + dt
        pred = y + dt * fy
        z = pred.copy()
        ok = False
        for _ in range(8):
            fz = f(t1, z)
            g = z - y - dt * fz
            J = np.eye(y.size) - dt * _jacobian(f, t1, z, fz)
            try:
                dz = np.linalg.solve(J, -g)
            except np.linalg.LinAlgError:
                break
            z = z + dz
            if not np.all(np.isfinite(z)):
                break
            if np.max(np.abs(dz) / (atol + rtol * np.abs(z))) < 1e-3:
                ok = True
                break
        if ok:
            err = 0.5 * np.max(np.abs(z - pred) / (atol + rtol * np.maximum(np.abs(z), np.abs(y))))
        if not ok or err > 1.0:
            dt *= 0.3 if not ok else max(0.2, 0.9 / np.sqrt(err))
            nrej += 1
            if dt < 1e-12:
                raise RuntimeError("step size underflow")
            continue
        t, y = t1, z
        fy = f(t, y)
        ts.append(t)
        ys.append(y.copy())
        nsteps += 1
        dt *= min(3.0, max(0.3, 0.9 / np.sqrt(max(err, 1e-10))))
    return np.array(ts), np.array(ys), dict(steps=nsteps, rejected=nrej)


def explicit_euler(f, y0, t_end, dt):
    n = int(round(t_end / dt))
    y = np.array(y0, float)
    ts = np.linspace(0.0, n * dt, n + 1)
    ys = np.empty((n + 1, y.size))
    ys[0] = y
    with np.errstate(all="ignore"):
        for i in range(n):
            y = y + dt * f(ts[i], y)
            ys[i + 1] = y
    return ts, ys


def oven_test(chem, T_oven, T0=301.15, t_end=7200.0, cell=None, rtol=1e-5):
    """Cell placed in an oven at T_oven. Returns time, state history and summary."""
    m = Module(chem, cell or Cell(), 1, T_amb=T_oven)
    t, y, info = backward_euler(m.rhs, initial_state(chem, T0), t_end, rtol=rtol)
    T = y[:, 0]
    runaway = bool(T.max() > T_oven + 100.0)
    t_run = float(t[np.argmax(T > T_oven + 100.0)]) if runaway else float("nan")
    return t, y, dict(runaway=runaway, T_peak=float(T.max()), t_runaway_s=t_run, **info)


def critical_oven_temperature(chem, lo=380.0, hi=520.0, tol=0.05, cell=None, t_end=14400.0):
    """Lowest oven temperature that leads to thermal runaway, by bisection."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if oven_test(chem, mid, cell=cell, t_end=t_end, rtol=1e-4)[2]["runaway"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --- scenarios -------------------------------------------------------------------------
def internal_short(chem, power, T_amb=298.15, t_end=7200.0, cell=None, rtol=1e-5):
    """One cell at ambient temperature with a constant internal heat source [W]."""
    m = Module(chem, cell or Cell(), 1, T_amb=T_amb, p_ext=lambda t: np.array([power]))
    t, y, info = backward_euler(m.rhs, initial_state(chem, T_amb), t_end, rtol=rtol)
    T = y[:, 0]
    runaway = bool(T.max() > 473.15 + 100.0)
    return t, y, dict(runaway=runaway, T_peak=float(T.max()))


def critical_short_power(chem, lo=1.0, hi=60.0, tol=0.02, **kw):
    """Smallest constant internal heating power that ends in thermal runaway, by bisection."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if internal_short(chem, mid, t_end=4 * 3600.0, rtol=1e-4, **kw)[2]["runaway"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def crossing_time(t, T, level):
    i = int(np.argmax(T > level))
    if T[i] <= level:
        return float("nan")
    return float(np.interp(level, T[i - 1:i + 1], t[i - 1:i + 1])) if i > 0 else float(t[0])


def propagation(chem, n=6, gap=1e-3, k_gap=0.03, heater=0.0, T_trigger=523.15, T_amb=298.15,
                t_end=3600.0, cell=None, G=None, rtol=1e-4, T_run=573.15):
    """Row of n cells. Trigger: cell 0 starts at T_trigger (default 250 C, above the onset
    of all three chemistries); alternatively it is heated with `heater` watts until it passes
    T_run, after which the heater is switched off. Neighbours are coupled by conduction through a gap
    (G = k_gap A_face / gap) and by radiation between facing areas A_face = height x diameter."""
    c = cell or Cell()
    A_face = c.height * 2 * c.radius
    G = k_gap * A_face / gap if G is None else G
    state = {"on": True}

    def p_ext(t):
        v = np.zeros(n)
        if state["on"]:
            v[0] = heater
        return v

    m = Module(chem, c, n, T_amb=T_amb, G=G, A_face=A_face, p_ext=p_ext)
    y0 = initial_state(chem, T_amb, n)
    if heater <= 0.0:
        y0[0] = T_trigger
        state["on"] = False
    # integrate in two legs so the heater switch is not inside a step
    tt, yy, _ = backward_euler(m.rhs, y0, t_end, rtol=rtol, dt_max=5.0)
    T0 = yy[:, 0]
    if heater > 0.0 and (T0 > T_run).any():
        k = int(np.argmax(T0 > T_run))
        t1, y1 = tt[:k + 1], yy[:k + 1]
        state["on"] = False
        t2, y2, _ = backward_euler(m.rhs, y1[-1], t_end - t1[-1], rtol=rtol, dt_max=5.0)
        tt = np.concatenate([t1, t1[-1] + t2[1:]])
        yy = np.vstack([y1, y2[1:]])
    T = yy[:, 0::NS]
    t_cross = [crossing_time(tt, T[:, i], T_run) for i in range(n)]
    return tt, T, dict(G=G, t_runaway_s=t_cross, n_runaway=int(np.sum(np.isfinite(t_cross))))
