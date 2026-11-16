"""The prediction experiment: formula vs simulation vs machine learning, scored against the reference.

For each random seed the 256 normal-range points are split into 56 test points, 150 training points and 50
calibration points. Training and calibration use noisy 'measurements' (2 % log-normal noise on the reference
runtime); the test scores use the reference itself. The 'outside' test set is the wide box (up to 121 points).
The formula is calibrated on two rated points of the reference (25 C, new cell, 50 % and 100 % load), as one
would read them from a datasheet; the simulation is not calibrated at all.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score

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
LEARNED_EXTRA = (PHYSICS, HYBRID)


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


def load_data(reference_path, model_points_path, column):
    """Reference points joined with the own model's runtime from `column` of the model-points file."""
    ref = rs.reference_points(reference_path)
    own = {ds.key(d): d for d in rs.load_json(model_points_path)}
    missing = [ds.key(d) for d in ref if ds.key(d) not in own]
    if missing:
        raise KeyError(f"{len(missing)} reference points have no model result, e.g. {missing[0]}")
    X, y, tag = rs.arrays(ref)
    sim = np.array([own[ds.key(d)][column] for d in ref])
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


def split(data, seed, n_train=N_TRAIN):
    """Test, training and calibration indices and the noisy measurements of one seed."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data.normal)
    test, pool = perm[:N_TEST], perm[N_TEST:]
    train, cal = pool[:n_train], pool[N_TRAIN:N_TRAIN + N_CAL]
    noisy = data.y * np.exp(NOISE * rng.standard_normal(len(data.y)))
    return test, train, cal, noisy


def learners(data, seed, names=None, sim_fun=None):
    mods = pr.make_models(seed)
    mods[PHYSICS] = pr.PhysicsFeatures()
    mods[HYBRID] = pr.Hybrid(sim_fun or sim_lookup(data))
    return {k: v for k, v in mods.items() if names is None or k in names}


def _record(data, pred, test, q=None, noisy=None):
    y, wide = data.y, data.wide
    def rel(idx):
        return (pred[idx] - y[idx]) / y[idx]
    rec = dict(mape_in=pr.mape(y[test], pred[test]), mape_wide=pr.mape(y[wide], pred[wide]),
               max_in=float(100 * np.max(np.abs(rel(test)))), max_wide=float(100 * np.max(np.abs(rel(wide)))),
               r2_in=float(r2_score(y[test], pred[test])), r2_wide=float(r2_score(y[wide], pred[wide])),
               bias_wide=float(100 * np.mean(rel(wide))))
    if q is not None:
        def inside(target, idx):
            return float(np.mean(np.abs(target[idx] - pred[idx]) <= q * pred[idx]))
        rec.update(conformal_q_percent=100 * q,
                   cover_in=inside(noisy, test), cover_wide=inside(noisy, wide),          # what conformal promises
                   cover_in_true=inside(y, test), cover_wide_true=inside(y, wide))       # against the noise-free truth
    return rec


def run_seed(data, seed, n_train=N_TRAIN, names=None, store=False, peukert=None):
    test, train, cal, noisy = split(data, seed, n_train)
    peukert = peukert or formula(data)
    preds = {FORMULA: peukert.predict(data.X), SIM: data.sim.copy()}
    res = {FORMULA: _record(data, preds[FORMULA], test), SIM: _record(data, preds[SIM], test)}
    fitted, qs = {}, {}
    for name, m in learners(data, seed, names).items():
        m.fit(data.X[train], noisy[train])
        preds[name] = m.predict(data.X)
        fitted[name] = m
        qs[name] = pr.conformal_quantile(noisy[cal], preds[name][cal], ALPHA)
        res[name] = _record(data, preds[name], test, qs[name], noisy)
    if "Gaussian process" in fitted:
        mu, sd = fitted["Gaussian process"].predict(data.X, return_std=True)
        res["Gaussian process"]["gp_cover_in"] = float(np.mean(np.abs(data.y[test] - mu[test]) <= 1.645 * sd[test]))
        res["Gaussian process"]["gp_cover_wide"] = float(np.mean(np.abs(data.y[data.wide] - mu[data.wide])
                                                                 <= 1.645 * sd[data.wide]))
    if store:
        return res, dict(preds=preds, q=qs, test=test, train=train, cal=cal)
    return res


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


def _seed_job(args):
    """run_seed with packed arguments (for a process pool)."""
    data, seed, n_train, names, store, peukert = args
    return run_seed(data, seed, n_train, names=names, store=store, peukert=peukert)


def run(data, n_seeds=20, learning_sizes=(5, 10, 20, 40, 80, 150), learning_seeds=10, log=print, mapper=map):
    """The whole experiment. `mapper` maps _seed_job over the seeds: the built-in map, or the map of a process
    pool; every seed is independent and deterministic, so the result does not depend on it."""
    peukert = formula(data)
    out = dict(noise=NOISE, n_train=N_TRAIN, n_cal=N_CAL, n_test_in=N_TEST, n_wide=int(len(data.wide)),
               peukert=dict(t_rated=peukert.t_rated, k=peukert.k))
    runs = list(mapper(_seed_job, [(data, s, N_TRAIN, None, True, peukert) for s in range(n_seeds)]))
    names = list(runs[0][0].keys())
    out["accuracy"] = {n: {m: [float(np.mean([r[0][n][m] for r in runs])), float(np.std([r[0][n][m] for r in runs]))]
                           for m in runs[0][0][n]} for n in names}
    for n in names:
        a = out["accuracy"][n]
        log(f"{n:32s} MAPE in {a['mape_in'][0]:6.2f}  out {a['mape_wide'][0]:7.2f}"
            + (f"  coverage in {a['cover_in'][0]:.2f} out {a['cover_wide'][0]:.2f}" if "cover_in" in a else ""))
    out["by_distance"] = by_distance(data, [r[1]["preds"] for r in runs], names)
    lc_names = ("Random forest", "Gradient boosting", "Neural network", "Gaussian process", PHYSICS, HYBRID)
    seeds = range(min(n_seeds, learning_seeds))
    results = list(mapper(_seed_job, [(data, s, n, lc_names, False, peukert) for n in learning_sizes for s in seeds]))
    out["learning_curve"] = {}
    for k, n_train in enumerate(learning_sizes):
        rr = results[k * len(seeds):(k + 1) * len(seeds)]
        out["learning_curve"][str(n_train)] = {n: dict(mape_in=float(np.mean([r[n]["mape_in"] for r in rr])),
                                                       mape_wide=float(np.mean([r[n]["mape_wide"] for r in rr])))
                                               for n in lc_names}
        log(f"N = {n_train}: " + ", ".join(f"{k} {v['mape_in']:.2f}/{v['mape_wide']:.1f}"
                                           for k, v in out["learning_curve"][str(n_train)].items()))
    st0 = runs[0][1]
    preds0 = {k: np.asarray(v).tolist() for k, v in st0["preds"].items()}
    out["seed0"] = dict(X=data.X.tolist(), y=data.y.tolist(), tag=data.tag.tolist(), test_idx=st0["test"].tolist(),
                        train_idx=st0["train"].tolist(), preds=preds0, conformal_q=st0["q"],
                        distance=pr.distance_outside(data.X).tolist())
    return out
