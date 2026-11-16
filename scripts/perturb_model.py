"""How much does the simulation's accuracy depend on having the reference's exact parameters?

The simulation and the reference share every parameter, which favours the simulation. Here one parameter group
at a time is set wrong (solid diffusivities, exchange currents, ohmic resistances by 30 %; the heat-transfer
coefficient by a factor of two) and the simulation is scored again against the reference, inside the normal
range and on the outside points.

Writes results/perturbation.json. About an hour on one core; --workers runs the cases in parallel.
"""
import argparse
import time
from multiprocessing import Pool

import _path  # noqa: F401
import numpy as np

from ups_battery_prediction import design as ds
from ups_battery_prediction import discharge as dc
from ups_battery_prediction import params as p
from ups_battery_prediction import predictors as pr
from ups_battery_prediction import results as rs

CASES = {
    "exact parameters": (dc.Scale(), p.H_COOL),
    "diffusivities x0.7": (dc.Scale(diffusivity=0.7), p.H_COOL),
    "diffusivities x1.3": (dc.Scale(diffusivity=1.3), p.H_COOL),
    "exchange currents x0.7": (dc.Scale(exchange=0.7), p.H_COOL),
    "exchange currents x1.3": (dc.Scale(exchange=1.3), p.H_COOL),
    "resistances x0.7": (dc.Scale(ohmic=0.7), p.H_COOL),
    "resistances x1.3": (dc.Scale(ohmic=1.3), p.H_COOL),
    "cooling h x0.5": (dc.Scale(), 0.5 * p.H_COOL),
    "cooling h x2": (dc.Scale(), 2.0 * p.H_COOL),
    "all worse (D, j0 x0.7, R x1.3)": (dc.Scale(0.7, 0.7, 1.3), p.H_COOL),
}


def points():
    pts = [d for d in rs.reference_points(_path.result("reference.json")) if d["tag"] in ("normal", "wide")]
    return rs.arrays(pts)


def case(name):
    scale, h = CASES[name]
    X, y, tag = points()
    t0 = time.time()
    r = dc.simulate(X[:, 0] + 273.15, X[:, 2], power=ds.P_RATED * X[:, 1], scale=scale, h=h)
    t = r["runtime_s"] / 60
    res = dict(inside=pr.mape(y[tag == "normal"], t[tag == "normal"]),
               outside=pr.mape(y[tag == "wide"], t[tag == "wide"]), n_nan=int(np.isnan(t).sum()),
               scale=dict(diffusivity=scale.diffusivity, exchange=scale.exchange, ohmic=scale.ohmic), h=h)
    print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()},
          f"[{time.time() - t0:.0f} s]", flush=True)
    return name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="cases run in parallel (each needs a few hundred MB)")
    args = ap.parse_args()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            out = dict(pool.map(case, list(CASES)))
    else:
        out = dict(map(case, CASES))
    out = {name: out[name] for name in CASES}
    X, y, tag = points()
    rs.save_json(dict(cases=out, n_inside=int((tag == "normal").sum()), n_outside=int((tag == "wide").sum())),
                  _path.result("perturbation.json"), indent=1)


if __name__ == "__main__":
    main()
