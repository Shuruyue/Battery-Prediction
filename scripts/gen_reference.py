"""Reference runtimes from PyBaMM (DFN + lumped thermal, O'Regan 2022) at every design point.

Resumable: results are appended to the output file after every point, and points already there are skipped.
Memory: each worker holds about 1.3 GB (mesh r = 160); keep workers x 1.5 GB well below the free RAM.
Each worker process builds the PyBaMM simulation once. Several hours of CPU time in total.

Usage: python scripts/gen_reference.py [--workers 4] [--out results/reference.json] [--tags normal,wide]
                                       [--thermal lumped|isothermal] [--h 10]
"""
import argparse
import os
import time
from multiprocessing import Pool

import _path  # noqa: F401

from ups_battery_prediction import params as p
from ups_battery_prediction import reference as ref

_SIM = None


def _init(thermal, h):
    global _SIM
    os.environ["OMP_NUM_THREADS"] = "1"
    _SIM = (thermal, h, ref.build_simulation(thermal=thermal, h=h))


def _job(d):
    thermal, h, sim = _SIM
    t0 = time.time()
    try:
        d.update(ref.run_point(sim, d["T_C"], d["load"], d["soh"]))
    except Exception as e:  # keep the failure in the record; the point is retried on the next run
        d.update(runtime_min=None, ok=False, termination="error: " + str(e)[:120])
    d.update(wall_s=time.time() - t0, mesh=ref.MESH, thermal=thermal, h=h)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=_path.result("reference.json"))
    ap.add_argument("--tags", default="", help="comma-separated tag prefixes to run (default: all)")
    ap.add_argument("--thermal", default=ref.THERMAL)
    ap.add_argument("--h", type=float, default=p.H_COOL)
    ap.add_argument("--skip", nargs="*", default=[], help="other result files whose points are not run again")
    ap.add_argument("--reverse", action="store_true", help="run the design from its end (a second, parallel job)")
    args = ap.parse_args()
    pts = [d for d in ref.design_points() if not args.tags or any(d["tag"].startswith(t) for t in args.tags.split(","))]
    done = [d for d in (ref.load_json(args.out) if os.path.exists(args.out) else []) if d.get("ok")]
    have = {ref.key(d) for d in done}
    for other in args.skip:
        have |= {ref.key(d) for d in ref.load_json(other) if d.get("ok")}
    todo = [d for d in (pts[::-1] if args.reverse else pts) if ref.key(d) not in have]
    print(f"{len(done)} done, {len(todo)} to run on {args.workers} workers "
          f"(mesh {ref.MESH}, {args.thermal}, h = {args.h})",
          flush=True)
    with Pool(args.workers, initializer=_init, initargs=(args.thermal, args.h)) as pool:
        for d in pool.imap_unordered(_job, todo):
            done.append(d)
            ref.save_json(done, args.out)
            print(f"{len(done)}/{len(pts)} {d['tag']} T={d['T_C']:.1f} load={d['load']:.2f} soh={d['soh']:.2f}: "
                  f"{d.get('runtime_min')} min, peak {d.get('T_peak_C')} C [{d['wall_s']:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
