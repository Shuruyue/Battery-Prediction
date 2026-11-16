"""Grid and time-step convergence of the hand-written model, and of PyBaMM's SPM on its default mesh.

1. Single-particle model, 0 C, 2C (10 A), isothermal: ours with 10 to 240 radial cells and PyBaMM's SPM with
   20 (its default) to 480 radial points. The Kirchhoff-transform faces make the difference at steep fronts.
2. The simulation tool (SPMe + lumped thermal) in its hardest corner of the design (0 C, 1.5 x rated power,
   state of health 0.7): radial cells, electrolyte cells and time step, one at a time.

Writes results/convergence.json. About twenty minutes on one core.
"""
import _path  # noqa: F401
import pybamm

from ups_battery_prediction import discharge as dc
from ups_battery_prediction import reference as ref

pybamm.set_logging_level("ERROR")
T_SPM, I_SPM = 273.15, 10.0


def spm_case():
    own = []
    for n in (10, 15, 20, 30, 60, 120, 240):
        r = dc.simulate(T_SPM, 1.0, current=I_SPM, model="SPM", thermal="isothermal", dt=0.25, n_r=n, film=False)
        own.append(dict(n_r=n, runtime_s=float(r["runtime_s"][0])))
        print("own SPM", own[-1], flush=True)
    pb = []
    for n in (20, 30, 60, 120, 240, 480):
        pv = pybamm.ParameterValues("ORegan2022")
        pv.update({"Ambient temperature [K]": T_SPM, "Initial temperature [K]": T_SPM})
        sim = pybamm.Simulation(pybamm.lithium_ion.SPM(), parameter_values=pv,
                                experiment=pybamm.Experiment([f"Discharge at {I_SPM} A until 2.5 V (1 second period)"]),
                                var_pts={"x_n": 10, "x_s": 10, "x_p": 10, "r_n": n, "r_p": n},
                                solver=pybamm.IDAKLUSolver(rtol=1e-9, atol=1e-9))
        pb.append(dict(n_r=n, runtime_s=float(sim.solve()["Time [s]"].entries[-1])))
        print("PyBaMM SPM", pb[-1], flush=True)
    return dict(T_K=T_SPM, current_A=I_SPM, own=own, pybamm=pb)


def tool_case():
    kw = dict(T_amb=273.15, soh=0.7, power=1.5 * ref.P_RATED, model="SPMe", thermal="lumped")
    base = dict(dt=1.0, n_r=30, n_x=(20, 10, 20))
    rows = []
    variations = ([dict(n_r=n) for n in (10, 15, 30, 60, 120)] + [dict(n_x=(n, n // 2, n)) for n in (10, 20, 40, 80)]
                  + [dict(dt=d) for d in (4.0, 2.0, 1.0, 0.5, 0.25)])
    for v in variations:
        opts = {**base, **v}
        r = dc.simulate(**kw, **opts)
        rows.append(dict(varied=list(v)[0], value=list(v.values())[0], runtime_s=float(r["runtime_s"][0]),
                         T_max_C=float(r["T_max_K"][0] - 273.15)))
        print("tool", rows[-1], flush=True)
    return dict(case=dict(T_C=0.0, load=1.5, soh=0.7), base=base, runs=rows)


def main():
    ref.save_json(dict(spm=spm_case(), tool=tool_case()), _path.result("convergence.json"), indent=1)


if __name__ == "__main__":
    main()
