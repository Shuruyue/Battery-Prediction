"""How much self-heating changes the runtime: rated load (18 W), new cell, at 0 C and 25 C, for several
heat-transfer coefficients h and for the isothermal limit; PyBaMM's DFN (the reference model) and ours.

Writes results/thermal_effect.json. About fifteen minutes on one core.
"""
import _path  # noqa: F401

from ups_battery_prediction import discharge as dc
from ups_battery_prediction import reference as ref

H_VALUES = (2.0, 5.0, 10.0, 25.0, 100.0)


def main():
    rows = []
    for T_C in (0.0, 25.0):
        for h in H_VALUES + (None,):
            thermal = "isothermal" if h is None else "lumped"
            sim = ref.build_simulation(thermal=thermal, h=10.0 if h is None else h)
            dfn = ref.run_point(sim, T_C, 1.0, 1.0)
            own = dc.simulate(T_C + 273.15, 1.0, power=ref.P_RATED, model="SPMe", thermal=thermal,
                              h=10.0 if h is None else h)
            rows.append(dict(T_C=T_C, h=h, dfn_min=dfn["runtime_min"], dfn_T_peak_C=dfn["T_peak_C"],
                             own_min=float(own["runtime_s"][0] / 60), own_T_peak_C=float(own["T_max_K"][0] - 273.15)))
            print(rows[-1], flush=True)
    ref.save_json(dict(load=1.0, soh=1.0, rows=rows), _path.result("thermal_effect.json"), indent=1)


if __name__ == "__main__":
    main()
