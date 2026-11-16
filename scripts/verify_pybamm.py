"""The hand-written model against PyBaMM, in two checks with fixed settings (no case is selected afterwards).

1. Code check: the same equations solved by both codes. PyBaMM's SPM, SPMe and SPMe + lumped thermal against
   ours, at constant current, with the electrolyte's chi and kappa_e evaluated at 1000 mol/m3 in both codes
   (PyBaMM's SPMe evaluates them at an x-average that its discretisation gets wrong; see discharge.py). Both
   codes on fine meshes. Voltage RMS (t > 1 s, up to 98 % of the shorter run), runtime and end temperature.
2. Model-form check: our SPM and SPMe, isothermal and lumped, against PyBaMM's DFN of the same thermal form
   (the model the reference uses), at constant power.

Writes results/verify_pybamm.json. About half an hour on one core.
"""
import time

import _path  # noqa: F401
import numpy as np
import pybamm

from ups_battery_prediction import discharge as dc
from ups_battery_prediction import params as p
from ups_battery_prediction import results as rs

pybamm.set_logging_level("ERROR")
CODE_CASES = [(25.0, 2.5), (25.0, 5.0), (25.0, 10.0), (0.0, 2.5), (0.0, 5.0), (-10.0, 2.5)]   # (T [C], current [A])
CODE_FORMS = [("SPM", "isothermal"), ("SPMe", "isothermal"), ("SPMe", "lumped")]
FORM_CASES = [(25.0, 9.0), (25.0, 18.0), (25.0, 27.0), (10.0, 18.0), (0.0, 9.0), (0.0, 18.0), (0.0, 27.0)]  # (T, W)
PB_MESH = {"x_n": 40, "x_s": 20, "x_p": 40, "r_n": 160, "r_p": 160}


def _const(f):
    return lambda c, T: f(1000.0 + 0.0 * np.asarray(c, float), T)


CONST = p.variant(chi=_const(p.chi), kappa_e=_const(p.kappa_e))


def pybamm_values(T, constant_electrolyte):
    pv = pybamm.ParameterValues("ORegan2022")
    pv.update({"Ambient temperature [K]": T, "Initial temperature [K]": T})
    if constant_electrolyte:
        tp, tdf = pv["Cation transference number"], pv["Thermodynamic factor"]
        kap = pv["Electrolyte conductivity [S.m-1]"]
        # chi = 2 (1 - t+) TDF: keep t+(c) for migration and put the concentration dependence of chi into TDF
        pv.update({"Thermodynamic factor": lambda c, T: tdf(1000.0, T) * (1 - tp(1000.0, T)) / (1 - tp(c, T)),
                   "Electrolyte conductivity [S.m-1]": lambda c, T: kap(1000.0, T)})
    return pv


def run_pybamm(model, thermal, T, step, constant_electrolyte, mesh=PB_MESH):
    m = getattr(pybamm.lithium_ion, model)(options={"thermal": thermal})
    sim = pybamm.Simulation(m, parameter_values=pybamm_values(T, constant_electrolyte), var_pts=mesh,
                            experiment=pybamm.Experiment([step + " until 2.5 V (5 second period)"]),
                            solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-9))
    sol = sim.solve()
    return (sol["Time [s]"].entries, sol["Voltage [V]"].entries, sol["X-averaged cell temperature [K]"].entries)


def code_check():
    rows = []
    for model, thermal in CODE_FORMS:
        for Tc, I in CODE_CASES:
            T = Tc + 273.15
            t0 = time.time()
            tr, vr, Tr = run_pybamm(model, thermal, T, f"Discharge at {I} A", True)
            r = dc.simulate(T, 1.0, current=I, model=model, thermal=thermal, dt=0.25, n_r=60, n_x=(40, 20, 40),
                            film=False, return_traces=True, prm=CONST)
            t, v = r["t"], r["V"][:, 0]
            m = (tr > 1.0) & (tr < 0.98 * min(tr[-1], r["runtime_s"][0]))
            err = np.interp(tr[m], t, v) - vr[m]
            row = dict(model=model, thermal=thermal, T_C=Tc, current_A=I, C_rate=I / p.Q_NOMINAL,
                       rms_mV=float(1e3 * np.sqrt(np.mean(err ** 2))), max_mV=float(1e3 * np.max(np.abs(err))),
                       runtime_own_s=float(r["runtime_s"][0]), runtime_pybamm_s=float(tr[-1]),
                       runtime_diff_percent=float(100 * (r["runtime_s"][0] / tr[-1] - 1)),
                       T_end_own_C=float(r["T_end_K"][0] - 273.15), T_end_pybamm_C=float(Tr[-1] - 273.15))
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                  f"[{time.time() - t0:.0f} s]", flush=True)
    return rows


def form_check():
    rows = []
    for thermal in ("isothermal", "lumped"):
        for Tc, P in FORM_CASES:
            T = Tc + 273.15
            row = dict(thermal=thermal, T_C=Tc, power_W=P)
            for model in ("DFN", "SPMe"):
                try:
                    row[f"pybamm_{model}_min"] = float(run_pybamm(model, thermal, T, f"Discharge at {P} W",
                                                                  False)[0][-1] / 60)
                except pybamm.SolverError as e:
                    row[f"pybamm_{model}_min"], row[f"pybamm_{model}_error"] = None, str(e)[:100]
            for model in ("SPM", "SPMe"):
                row[f"own_{model}_min"] = float(dc.simulate(T, 1.0, power=P, model=model, thermal=thermal, dt=0.5,
                                                            n_r=40, film=False)["runtime_s"][0] / 60)
            rows.append(row)
            print(row, flush=True)
    return rows


def main():
    out = dict(pybamm_version=pybamm.__version__, pybamm_mesh=PB_MESH, code_check=code_check(), form_check=form_check())
    rs.save_json(out, _path.result("verify_pybamm.json"), indent=1)


if __name__ == "__main__":
    main()
