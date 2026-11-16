"""Build web/ups-battery.html from web/src/: inline the chart code, the thermal-runaway model and the data block.

The data block holds, on the regular grid of results/model_grid.json, the runtime of the simulation and of every
learned model (trained on seed 0 of the experiment, the same split and noise as in results/ml_results.json),
their conformal half-width factors, the Peukert constants and every reference point.

Usage: python scripts/build_web_page.py
"""
import json
import warnings
from pathlib import Path

import _path  # noqa: F401
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.exceptions import ConvergenceWarning

from ups_battery_prediction import experiment as ex
from ups_battery_prediction import predictors as pr
from ups_battery_prediction import reference as ref

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ROOT = Path(_path.ROOT)
SRC = ROOT / "web" / "src"
PAGE = ROOT / "web" / "ups-battery.html"
LEARNED = {"Gradient boosting": "Gradient boosting", "Neural network": "Neural network",
           "Gaussian process": "Gaussian process", "Physics features": ex.PHYSICS, "Simulation + data": ex.HYBRID}


def grid_data():
    g = ref.load_json(_path.result("model_grid.json"))
    T, L, H = (np.array(g[k]) for k in ("T_C", "load", "soh"))
    G = np.array(g["runtime_min"])
    interp = RegularGridInterpolator((T, L, H[::-1]), G[:, :, ::-1], bounds_error=False, fill_value=None)
    data = ex.load_data(_path.result("reference.json"), _path.result("model_points.json"), "SPMe_lumped")
    sim_fun = ex.sim_lookup(data, fallback=interp)
    TT, LL, HH = np.meshgrid(T, L, H, indexing="ij")
    XG = np.column_stack([TT.ravel(), LL.ravel(), HH.ravel()])
    _, train, cal, noisy = ex.split(data, 0)
    peukert = ex.formula(data)
    models, q = {"Simulation": np.round(G.ravel(), 3).tolist()}, {}
    fitted = ex.learners(data, 0, names=set(LEARNED.values()), sim_fun=sim_fun)
    for short, name in LEARNED.items():
        m = fitted[name].fit(data.X[train], noisy[train])
        models[short] = np.round(m.predict(XG), 3).tolist()
        q[short] = pr.conformal_quantile(noisy[cal], m.predict(data.X[cal]), ex.ALPHA)
        span = f"{min(models[short]):.1f}-{max(models[short]):.1f} min"
        print(f"{short}: grid {span}, conformal q {100 * q[short]:.1f} %")
    reference = [dict(T=round(float(x[0]), 4), load=round(float(x[1]), 4), soh=round(float(x[2]), 4),
                      t=round(float(t), 3),
                      tag=str(tag)) for x, t, tag in zip(data.X, data.y, data.tag, strict=True)]
    th = {(r["T_C"], r["h"]): r for r in ref.load_json(_path.result("thermal_effect.json"))["rows"]}
    cold_iso, cold = th[(0.0, None)], th[(0.0, 10.0)]
    warm_iso, warm = th[(25.0, None)], th[(25.0, 10.0)]
    about_heat = (f"At rated load the reference lasts {warm['dfn_min']:.1f} min at 25 °C and "
                  f"{cold['dfn_min']:.1f} min at 0 °C "
                  f"when the cell heats itself (cooling 10 W/(m² K)); held at the room temperature it would last "
                  f"{warm_iso['dfn_min']:.1f} and {cold_iso['dfn_min']:.1f} min. The cell starting at 0 °C ends at "
                  f"{cold['dfn_T_peak_C']:.0f} °C: its own losses warm it, and a warmer cell has lower losses. "
                  f"How much this helps depends on how well the cabinet removes heat, which this model assumes.")
    return dict(T=T.tolist(), load=L.tolist(), soh=H.tolist(), normal=ref.NORMAL,
                peukert=dict(t_rated=peukert.t_rated, k=peukert.k),
                models=models, q=q, T_peak=np.round(np.array(g["T_peak_C"]).ravel(), 2).tolist(), reference=reference,
                about_heat=about_heat)


def build(data):
    page = (SRC / "ups-battery.src.html").read_text(encoding="utf-8")
    parts = {"/*__CHART__*/": (SRC / "chart.js").read_text(encoding="utf-8"),
             "/*__MODEL__*/": (SRC / "runaway_model.js").read_text(encoding="utf-8"),
             "/*__DATA__*/": "window.RUNTIME_DATA = " + json.dumps(data, separators=(",", ":")) + ";"}
    for token, text in parts.items():
        assert page.count(token) == 1, token
        page = page.replace(token, text)
    PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f"{PAGE.relative_to(ROOT)}: {len(page) // 1024} kB")


if __name__ == "__main__":
    build(grid_data())
