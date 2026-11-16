"""The simulation tool (SPMe + lumped thermal) on a regular grid: 0-45 C, 20-150 % of rated power, state of
health 1.0-0.7. Used by the runtime map and by the interactive page. Writes results/model_grid.json."""
import time

import _path  # noqa: F401
import numpy as np

from ups_battery_prediction import discharge as dc
from ups_battery_prediction import reference as ref

T_GRID = np.arange(0.0, 45.01, 2.5)
LOAD_GRID = np.round(np.arange(0.2, 1.501, 0.05), 2)
SOH_GRID = np.array([1.0, 0.9, 0.8, 0.7])


def main():
    TT, LL, HH = np.meshgrid(T_GRID, LOAD_GRID, SOH_GRID, indexing="ij")
    t0 = time.time()
    r = dc.simulate(TT.ravel() + 273.15, HH.ravel(), power=ref.P_RATED * LL.ravel(), model="SPMe", thermal="lumped")
    rt = (r["runtime_s"] / 60).reshape(TT.shape)
    tp = (r["T_max_K"] - 273.15).reshape(TT.shape)
    ref.save_json(dict(T_C=T_GRID.tolist(), load=LOAD_GRID.tolist(), soh=SOH_GRID.tolist(), runtime_min=rt.tolist(),
                       T_peak_C=tp.tolist()), _path.result("model_grid.json"))
    print(rt.shape, f"{time.time() - t0:.0f} s", float(np.nanmin(rt)), float(np.nanmax(rt)), int(np.isnan(rt).sum()))


if __name__ == "__main__":
    main()
