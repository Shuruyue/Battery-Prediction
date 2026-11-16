"""The hand-written model at every design point of the reference, in three forms:

  SPMe_lumped      the simulation tool of the experiment (electrolyte and self-heating included)
  SPMe_isothermal  the same without self-heating (cell held at the ambient temperature)
  SPM_isothermal   single-particle model only, isothermal (the form of the earlier version of this project)

Writes results/model_points.json. About ten minutes.
"""
import time

import _path  # noqa: F401
import numpy as np

from ups_battery_prediction import discharge as dc
from ups_battery_prediction import reference as ref

FORMS = {"SPMe_lumped": ("SPMe", "lumped"), "SPMe_isothermal": ("SPMe", "isothermal"),
         "SPM_isothermal": ("SPM", "isothermal")}


def main():
    pts = ref.design_points()
    T = np.array([d["T_C"] for d in pts]) + 273.15
    L = np.array([d["load"] for d in pts])
    H = np.array([d["soh"] for d in pts])
    for name, (model, thermal) in FORMS.items():
        t0 = time.time()
        r = dc.simulate(T, H, power=ref.P_RATED * L, model=model, thermal=thermal, dt=1.0, n_r=30)
        for d, t, e, tp in zip(pts, r["runtime_s"], r["energy_Wh"], r["T_max_K"], strict=True):
            d[name] = float(t / 60)
            d[name + "_energy_Wh"] = float(e)
            d[name + "_T_peak_C"] = float(tp - 273.15)
        rt = r["runtime_s"] / 60
        print(f"{name}: {time.time() - t0:.0f} s, runtime {np.nanmin(rt):.2f}-{np.nanmax(rt):.2f} min, "
              f"{int(np.isnan(rt).sum())} without result", flush=True)
    ref.save_json(pts, _path.result("model_points.json"))


if __name__ == "__main__":
    main()
