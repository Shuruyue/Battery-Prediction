"""Mesh audit of the reference: the hardest corners of the design on successively finer PyBaMM meshes.

Writes results/reference_mesh_audit.json. The mesh of the reference (reference.MESH) is the coarsest one whose
runtimes agree with the finest within 0.2 % at every audited point.

Usage: python scripts/audit_reference_mesh.py [--workers 8]
"""
import argparse
import os
import time
from multiprocessing import Pool

import _path  # noqa: F401

from ups_battery_prediction import reference as ref

POINTS = [(0.0, 1.5, 0.7), (0.0, 1.5, 1.0), (0.0, 1.0, 1.0), (10.0, 1.5, 0.7), (45.0, 1.5, 0.7), (25.0, 0.2, 1.0)]
MESHES = [dict(x_n=x, x_s=x // 2, x_p=x, r_n=r, r_p=r) for x in (20, 40) for r in (40, 80, 160, 320)]


def job(args):
    mesh, (T_C, load, soh) = args
    t0 = time.time()
    try:
        out = ref.run_point(ref.build_simulation(mesh), T_C, load, soh)
    except Exception as e:  # solver failure is a result too
        out = dict(ok=False, termination="error: " + str(e)[:120])
    out.update(mesh=mesh, T_C=T_C, load=load, soh=soh, wall_s=time.time() - t0)
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(args.workers) as pool:
        rows = pool.map(job, [(m, pt) for m in MESHES for pt in POINTS], chunksize=1)
    ref.save_json(dict(points=POINTS, meshes=MESHES, runs=rows), _path.result("reference_mesh_audit.json"), indent=1)


if __name__ == "__main__":
    main()
