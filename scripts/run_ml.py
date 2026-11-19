"""Formula vs simulation vs machine learning, scored against the reference (see experiment.py).

Writes results/ml_results.json. A minute or two on one core (the Gaussian process and the network dominate);
--workers spreads the seeds over processes (each holds a few hundred MB). The result does not depend on it.
Usage: python scripts/run_ml.py [--seeds 20] [--workers 1]
"""
import argparse
import os
import warnings
from multiprocessing import Pool

import _path  # noqa: F401
from sklearn.exceptions import ConvergenceWarning

from battery_prediction import experiment as ex
from battery_prediction import results as rs


def _quiet():
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    _quiet()
    data = ex.load_data(_path.result("reference.json"), _path.result("model_points.json"))
    print(f"{len(data.y)} reference points: {len(data.normal)} normal, {len(data.wide)} outside", flush=True)

    def log(s):
        print(s, flush=True)

    if args.workers > 1:
        with Pool(args.workers, initializer=_quiet) as pool:
            out = ex.run(data, n_seeds=args.seeds, log=log, mapper=pool.imap)
    else:
        out = ex.run(data, n_seeds=args.seeds, log=log)
    rs.save_json(out, _path.result("ml_results.json"))


if __name__ == "__main__":
    main()
