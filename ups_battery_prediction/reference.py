"""The reference ("reality") and the design of the experiment.

The reference is the Doyle-Fuller-Newman model with a lumped thermal model, solved by PyBaMM with the same
parameter set (O'Regan 2022): the cell starts at the ambient temperature and discharges at constant power,
cooled by h = 10 W/(m2 K) over its surface. It plays the part of the real battery: the training
'measurements' of the machine-learning tools come from it, and every predictor is scored against it.

Design: 256 points in the normal operating range (Sobol), four one-at-a-time sweeps, and up to 128 points of
a wider box (Sobol) that fall outside the normal range. Loads are fractions of the rated 18 W per cell.
"""
import json
import os

import numpy as np

from . import params as p
from .ageing import cell_state

P_RATED = 18.0                       # W per cell: a 56-cell, 1 kW UPS at rated load
NORMAL = dict(T=(18.0, 28.0), load=(0.3, 0.8), soh=(0.9, 1.0))
WIDE = dict(T=(0.0, 45.0), load=(0.2, 1.5), soh=(0.7, 1.0))
MESH = dict(x_n=20, x_s=10, x_p=20, r_n=160, r_p=160)   # chosen by scripts/audit_reference_mesh.py
THERMAL = "lumped"


def key(d):
    """Identity of a design point."""
    return d["tag"], round(d["T_C"], 6), round(d["load"], 6), round(d["soh"], 6)


def in_normal_range(T_C, load, soh):
    return ((NORMAL["T"][0] <= T_C <= NORMAL["T"][1]) and (NORMAL["load"][0] <= load <= NORMAL["load"][1])
            and soh >= NORMAL["soh"][0])


def design_points():
    from scipy.stats import qmc
    pts = []

    def add(tag, T_C, load, soh):
        pts.append(dict(tag=tag, T_C=float(T_C), load=float(load), soh=float(soh)))

    for u in qmc.Sobol(3, seed=7).random(256):
        add("normal", NORMAL["T"][0] + 10 * u[0], NORMAL["load"][0] + 0.5 * u[1], NORMAL["soh"][0] + 0.1 * u[2])
    for T in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
        add("sweep_T_0.6", T, 0.6, 1.0)
        add("sweep_T_1.0", T, 1.0, 1.0)
    for L in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        add("sweep_load", 25, L, 1.0)
    for h in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        add("sweep_soh", 25, 0.6, h)
    for u in qmc.Sobol(3, seed=11).random(128):
        T, L, h = WIDE["T"][0] + 45 * u[0], WIDE["load"][0] + 1.3 * u[1], WIDE["soh"][0] + 0.3 * u[2]
        if not in_normal_range(T, L, h):
            add("wide", T, L, h)
    return pts


# --- PyBaMM ---------------------------------------------------------------------------------------------------
def build_simulation(mesh=None, thermal=THERMAL, h=p.H_COOL, rtol=1e-7):
    """A PyBaMM DFN simulation whose operating point is given by inputs (see run_point)."""
    import pybamm
    pybamm.set_logging_level("ERROR")
    options = {"operating mode": "power", "contact resistance": "true", "thermal": thermal}
    pv = pybamm.ParameterValues("ORegan2022")
    pv.update({"Ambient temperature [K]": "[input]", "Initial temperature [K]": "[input]",
               "Power function [W]": "[input]", "Contact resistance [Ohm]": "[input]",
               "Initial concentration in negative electrode [mol.m-3]": "[input]",
               "Initial concentration in positive electrode [mol.m-3]": "[input]",
               "Total heat transfer coefficient [W.m-2.K-1]": h}, check_already_exists=False)
    return pybamm.Simulation(pybamm.lithium_ion.DFN(options=options), parameter_values=pv,
                             var_pts=dict(mesh or MESH), solver=pybamm.IDAKLUSolver(rtol=rtol, atol=rtol))


def run_point(sim, T_C, load, soh):
    """Discharge of one design point; returns runtime (min), end voltage, peak temperature (C), termination."""
    st = cell_state(soh)
    T = T_C + 273.15
    inputs = {"Ambient temperature [K]": T, "Initial temperature [K]": T, "Power function [W]": P_RATED * load,
              "Contact resistance [Ohm]": st.r_contact,
              "Initial concentration in negative electrode [mol.m-3]": st.x0 * p.C_N_MAX,
              "Initial concentration in positive electrode [mol.m-3]": st.y0 * p.C_P_MAX}
    t_max = 3.0 * 3600 * 18.2 / (P_RATED * load)
    sol = sim.solve(np.linspace(0, t_max, 400), inputs=inputs)
    t = sol["Time [s]"].entries
    v = sol["Voltage [V]"].entries
    T_cell = sol["X-averaged cell temperature [K]"].entries
    return dict(runtime_min=float(t[-1] / 60), v_end=float(v[-1]), T_peak_C=float(T_cell.max() - 273.15),
                termination=str(sol.termination), ok=bool(abs(v[-1] - p.V_MIN) < 0.02))


# --- results --------------------------------------------------------------------------------------------------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.read().replace("NaN", "null"))


def save_json(obj, path, indent=0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def reference_points(path):
    """Design points of the reference with a valid runtime, in design order."""
    order = {key(d): i for i, d in enumerate(design_points())}
    ref = [d for d in load_json(path) if d.get("ok")]
    return sorted(ref, key=lambda d: order.get(key(d), 10 ** 6))


def arrays(points, column="runtime_min"):
    X = np.array([[d["T_C"], d["load"], d["soh"]] for d in points])
    y = np.array([d[column] for d in points])
    tag = np.array([d["tag"] for d in points])
    return X, y, tag
