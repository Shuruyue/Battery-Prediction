"""Formula vs simulation vs machine learning, scored against the reference (see experiment.py).

Also scores the three forms of the hand-written model (results/model_points.json) on all normal-range and all
outside points, which shows what leaving out self-heating or the electrolyte costs.

Writes results/ml_results.json. About an hour on one core (the Gaussian process and the network dominate);
--workers spreads the seeds over processes (each holds a few hundred MB). The result does not depend on it.
Usage: python scripts/run_ml.py [--seeds 20] [--workers 1]
"""
import argparse
import os
import warnings
from multiprocessing import Pool

import _path  # noqa: F401
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ups_battery_prediction import experiment as ex
from ups_battery_prediction import predictors as pr
from ups_battery_prediction import reference as ref

FORMS = ("SPMe_lumped", "SPMe_isothermal", "SPM_isothermal")


def _quiet():
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    _quiet()
    data = ex.load_data(_path.result("reference.json"), _path.result("model_points.json"), FORMS[0])
    print(f"{len(data.y)} reference points: {len(data.normal)} normal, {len(data.wide)} outside", flush=True)

    def log(s):
        print(s, flush=True)

    if args.workers > 1:
        with Pool(args.workers, initializer=_quiet) as pool:
            out = ex.run(data, n_seeds=args.seeds, log=log, mapper=pool.imap)
    else:
        out = ex.run(data, n_seeds=args.seeds, log=log)
    forms = {}
    for col in FORMS:
        d = ex.load_data(_path.result("reference.json"), _path.result("model_points.json"), col)
        rel = (d.sim - d.y) / d.y
        forms[col] = dict(mape_normal=pr.mape(d.y[d.normal], d.sim[d.normal]),
                          mape_wide=pr.mape(d.y[d.wide], d.sim[d.wide]),
                          max_wide=float(100 * np.max(np.abs(rel[d.wide]))),
                          bias_wide=float(100 * np.mean(rel[d.wide])))
        print(col, {k: round(v, 2) for k, v in forms[col].items()}, flush=True)
    out["model_forms"] = forms
    ref.save_json(out, _path.result("ml_results.json"))


if __name__ == "__main__":
    main()
