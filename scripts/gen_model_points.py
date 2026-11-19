"""The simulation tool (SPMe + lumped thermal, parameters design.SIM_SCALE) at every design point of the reference.

Writes results/model_points.json. About five minutes.
"""
import time

import _path  # noqa: F401
import numpy as np

from battery_prediction import design as ds
from battery_prediction import discharge as dc
from battery_prediction import results as rs


def main():
    pts = ds.design_points()
    T = np.array([d["T_C"] for d in pts]) + 273.15
    L = np.array([d["load"] for d in pts])
    H = np.array([d["soh"] for d in pts])
    t0 = time.time()
    r = dc.simulate(T, H, power=ds.P_RATED * L, model="SPMe", thermal="lumped", scale=dc.Scale(**ds.SIM_SCALE))
    for d, t, tp in zip(pts, r["runtime_s"], r["T_max_K"], strict=True):
        d["runtime_min"] = float(t / 60)
        d["T_peak_C"] = float(tp - 273.15)
    rt = r["runtime_s"] / 60
    print(f"{time.time() - t0:.0f} s, runtime {np.nanmin(rt):.2f}-{np.nanmax(rt):.2f} min, "
          f"{int(np.isnan(rt).sum())} without result", flush=True)
    rs.save_json(pts, _path.result("model_points.json"))


if __name__ == "__main__":
    main()
