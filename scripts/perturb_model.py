"""How much does the simulation's accuracy depend on its parameters, and can data fix wrong ones?

The simulation tool of the experiment uses a pessimistic parameter set (design.SIM_SCALE). Here it is run with
the reference's exact parameters (a best case: the two models then share every parameter), with one parameter
group at a time wrong (solid diffusivities, exchange currents, ohmic resistances by 30 %; the heat-transfer
coefficient by a factor of two) and with the tool's own set. Each is scored against the reference on the same 20
splits as run_ml.py, alone and with the data correction ('simulation + data') trained on top of it.

Writes results/perturbation.json. Each case simulates all 418 points (about six minutes on one core);
--workers runs the cases in parallel (each needs a few hundred MB).
Usage: python scripts/perturb_model.py [--workers 1] [--seeds 20]
"""
import argparse
import os
import time
import warnings
from multiprocessing import Pool

import _path  # noqa: F401
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from battery_prediction import design as ds
from battery_prediction import discharge as dc
from battery_prediction import experiment as ex
from battery_prediction import params as p
from battery_prediction import results as rs

TOOL = ex.TOOL_CASE
CASES = {
    ex.EXACT_CASE: (dc.Scale(), p.H_COOL),
    "diffusivities x0.7": (dc.Scale(diffusivity=0.7), p.H_COOL),
    "diffusivities x1.3": (dc.Scale(diffusivity=1.3), p.H_COOL),
    "exchange currents x0.7": (dc.Scale(exchange=0.7), p.H_COOL),
    "exchange currents x1.3": (dc.Scale(exchange=1.3), p.H_COOL),
    "resistances x0.7": (dc.Scale(ohmic=0.7), p.H_COOL),
    "resistances x1.3": (dc.Scale(ohmic=1.3), p.H_COOL),
    "cooling h x0.5": (dc.Scale(), 0.5 * p.H_COOL),
    "cooling h x2": (dc.Scale(), 2.0 * p.H_COOL),
    TOOL: (dc.Scale(**ds.SIM_SCALE), p.H_COOL),
}


def _quiet():
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load():
    return ex.load_data(_path.result("reference.json"), _path.result("model_points.json"))


def simulate_case(name):
    """Runtime [min] of the simulation with the parameters of `name` at every reference point."""
    _quiet()
    data = load()
    scale, h = CASES[name]
    t0 = time.time()
    r = dc.simulate(data.X[:, 0] + 273.15, data.X[:, 2], power=ds.P_RATED * data.X[:, 1], scale=scale, h=h)
    print(f"{name}: {len(data.X)} points in {time.time() - t0:.0f} s", flush=True)
    return name, r["runtime_s"] / 60


def score(data, sim, n_seeds, peukert):
    """The simulation alone and corrected by data, on the splits of run_ml.py."""
    d = ex.Data(data.X, data.y, data.tag, np.asarray(sim, float))
    runs = [ex.run_seed(d, s, names=(ex.HYBRID,), peukert=peukert) for s in range(n_seeds)]
    return ex.mean_std(runs, (ex.SIM, ex.HYBRID))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    _quiet()
    if args.workers > 1:
        with Pool(args.workers, initializer=_quiet) as pool:
            sims = dict(pool.map(simulate_case, list(CASES)))
    else:
        sims = dict(map(simulate_case, CASES))
    data = load()
    gap = float(np.max(np.abs(sims[TOOL] / data.sim - 1)))
    print(f"{TOOL} against model_points.json: largest relative difference {gap:.1e}", flush=True)
    peukert = ex.formula(data)
    out = {}
    for name in CASES:
        out[name] = dict(n_nan=int(np.isnan(sims[name]).sum()), accuracy=score(data, sims[name], args.seeds, peukert))
        a = out[name]["accuracy"]
        print(f"{name:40s} alone in {a[ex.SIM]['mape_in'][0]:5.2f} out {a[ex.SIM]['mape_wide'][0]:5.2f} | "
              f"+ data in {a[ex.HYBRID]['mape_in'][0]:5.2f} out {a[ex.HYBRID]['mape_wide'][0]:5.2f} "
              f"coverage out {a[ex.HYBRID]['cover_wide'][0]:.2f}", flush=True)
    rs.save_json(dict(n_seeds=args.seeds, cases=out), _path.result("perturbation.json"), indent=1)


if __name__ == "__main__":
    main()
