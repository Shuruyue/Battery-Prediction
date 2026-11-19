"""The prediction experiment: formula vs simulation vs machine learning, scored against the reference.

For each random seed the 256 normal-range points are split into 56 test points, 150 training points and 50
calibration points. Training and calibration use noisy 'measurements' (2 % log-normal noise on the reference
runtime); the test scores use the reference itself. The 'outside' test set is the wide box (up to 121 points).
The formula is calibrated on two rated points of the reference (25 C, new cell, 50 % and 100 % load), as one
would read them from a datasheet; the simulation is not calibrated at all.
"""
from dataclasses import dataclass

import numpy as np

from . import design as ds
from . import predictors as pr
from . import results as rs

NOISE = 0.02
N_TEST, N_TRAIN, N_CAL = 56, 150, 50
ALPHA = 0.1
BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 99.0))
SIM = "Simulation (own model)"
FORMULA = "Formula (Peukert)"
PHYSICS = "Physics features + linear fit"
HYBRID = "Simulation + data correction"
ML = ("Gradient boosting", "Neural network", "Gaussian process")
EXACT_CASE = "exact parameters"                        # two cases of scripts/perturb_model.py
TOOL_CASE = "the tool's set (D, j0 x0.7, R x1.3)"


@dataclass
class Data:
    X: np.ndarray        # (n, 3): T [C], load, soh
    y: np.ndarray        # reference runtime, min
    tag: np.ndarray
    sim: np.ndarray      # own-model runtime at the same points, min

    @property
    def normal(self):
        return np.where(self.tag == "normal")[0]

    @property
    def wide(self):
        return np.where(self.tag == "wide")[0]

    def sweep(self, name):
        return np.where(self.tag == name)[0]


def load_data(reference_path, model_points_path):
    """Reference points joined with the own model's runtime at the same points."""
    ref = rs.reference_points(reference_path)
    own = {ds.key(d): d for d in rs.load_json(model_points_path)}
    missing = [ds.key(d) for d in ref if ds.key(d) not in own]
    if missing:
        raise KeyError(f"{len(missing)} reference points have no model result, e.g. {missing[0]}")
    X, y, tag = rs.arrays(ref)
    sim = np.array([own[ds.key(d)]["runtime_min"] for d in ref])
    return Data(X, y, tag, sim)


def sim_lookup(data, fallback=None):
    """The simulation as a function of X: exact values at design points, `fallback(X)` elsewhere."""
    table = {tuple(np.round(x, 6)): s for x, s in zip(data.X, data.sim, strict=True)}

    def f(Z):
        Z = np.atleast_2d(Z)
        out = np.array([table.get(tuple(np.round(z, 6)), np.nan) for z in Z])
        if np.isnan(out).any():
            if fallback is None:
                raise KeyError("simulation result missing for a requested point")
            out[np.isnan(out)] = fallback(Z[np.isnan(out)])
        return out
    return f


def formula(data):
    sl = data.sweep("sweep_load")
    at = {round(float(x[1]), 6): float(t) for x, t in zip(data.X[sl], data.y[sl], strict=True)}
    return pr.Peukert().fit_two_points(0.5, at[0.5], 1.0, at[1.0])


def split(data, seed):
    """Test, training and calibration indices and the noisy measurements of one seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.normal)
    test, pool = perm[:N_TEST], perm[N_TEST:]
    train, cal = pool[:N_TRAIN], pool[N_TRAIN:N_TRAIN + N_CAL]
    noisy = data.y * np.exp(NOISE * rng.standard_normal(len(data.y)))
    return test, train, cal, noisy


def learners(data, seed, names=None, sim_fun=None):
    mods = pr.make_models(seed)
    mods[PHYSICS] = pr.PhysicsFeatures()
    mods[HYBRID] = pr.Hybrid(sim_fun or sim_lookup(data))
    return {k: v for k, v in mods.items() if names is None or k in names}


def _record(data, pred, test, q=None, noisy=None):
    """Mean absolute percentage error inside (test points) and outside; with a conformal factor q, the share of
    noisy measurements inside the interval pred (1 +- q), which is what split conformal promises."""
    y, wide = data.y, data.wide
    rec = dict(mape_in=pr.mape(y[test], pred[test]), mape_wide=pr.mape(y[wide], pred[wide]))
    if q is not None:
        def inside(idx):
            return float(np.mean(np.abs(noisy[idx] - pred[idx]) <= q * pred[idx]))
        rec.update(cover_in=inside(test), cover_wide=inside(wide))
    return rec


def run_seed(data, seed, names=None, store=False, peukert=None):
    test, train, cal, noisy = split(data, seed)
    peukert = peukert or formula(data)
    preds = {FORMULA: peukert.predict(data.X), SIM: data.sim.copy()}
    res = {FORMULA: _record(data, preds[FORMULA], test), SIM: _record(data, preds[SIM], test)}
    for name, m in learners(data, seed, names).items():
        m.fit(data.X[train], noisy[train])
        preds[name] = m.predict(data.X)
        q = pr.conformal_quantile(noisy[cal], preds[name][cal], ALPHA)
        res[name] = _record(data, preds[name], test, q, noisy)
    return (res, preds) if store else res


def by_distance(data, preds_per_seed, names):
    """Error of each predictor against the distance of an outside point from the training box, pooled over seeds."""
    d = pr.distance_outside(data.X[data.wide])
    y = data.y[data.wide]
    out = {}
    for name in names:
        errs = np.stack([np.abs(p[name][data.wide] - y) / y for p in preds_per_seed])    # (seeds, points)
        out[name] = []
        for lo, hi in BINS:
            m = (d >= lo) & (d < hi)
            out[name].append(dict(bin=[lo, hi], n=int(m.sum()),
                                  mape=float(100 * errs[:, m].mean()) if m.any() else None))
    return out


def mean_std(runs, names):
    """Mean and standard deviation over seeds of every metric of every predictor."""
    return {n: {m: [float(np.mean([r[n][m] for r in runs])), float(np.std([r[n][m] for r in runs]))]
                for m in runs[0][n]} for n in names}


def _seed_job(args):
    """run_seed with packed arguments (for a process pool)."""
    data, seed, peukert = args
    return run_seed(data, seed, store=True, peukert=peukert)


def run(data, n_seeds=20, log=print, mapper=map):
    """The whole experiment. `mapper` maps _seed_job over the seeds: the built-in map, or the map of a process
    pool; every seed is independent and deterministic, so the result does not depend on it."""
    peukert = formula(data)
    out = dict(noise=NOISE, n_train=N_TRAIN, n_cal=N_CAL, n_test_in=N_TEST, n_wide=int(len(data.wide)),
               peukert=dict(t_rated=peukert.t_rated, k=peukert.k))
    runs = list(mapper(_seed_job, [(data, s, peukert) for s in range(n_seeds)]))
    names = list(runs[0][0].keys())
    out["accuracy"] = mean_std([r[0] for r in runs], names)
    for n in names:
        a = out["accuracy"][n]
        log(f"{n:32s} MAPE in {a['mape_in'][0]:6.2f}  out {a['mape_wide'][0]:7.2f}"
            + (f"  coverage in {a['cover_in'][0]:.2f} out {a['cover_wide'][0]:.2f}" if "cover_in" in a else ""))
    out["by_distance"] = by_distance(data, [r[1] for r in runs], names)
    out["seed0"] = dict(X=data.X.tolist(), y=data.y.tolist(), tag=data.tag.tolist(),
                        preds={k: np.asarray(v).tolist() for k, v in runs[0][1].items()})
    return out
