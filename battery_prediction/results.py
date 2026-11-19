"""Reading and writing results/*.json, and the reference points as arrays."""
import json
import os

import numpy as np

from .design import design_points, key


def load_json(path):
    """Read a results file. NaN (a run without a result) is read back as a float NaN."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, indent=0):
    """Write a results file atomically (a crash leaves the previous file intact)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def reference_points(path):
    """Design points of the reference with a valid runtime, in design order."""
    order = {key(d): i for i, d in enumerate(design_points())}
    ref = [d for d in load_json(path) if d.get("ok")]
    return sorted(ref, key=lambda d: order.get(key(d), 10 ** 6))


def arrays(points, column="runtime_min"):
    """(X, y, tag): X has the columns ambient temperature [C], load, state of health."""
    X = np.array([[d["T_C"], d["load"], d["soh"]] for d in points])
    y = np.array([d[column] for d in points])
    tag = np.array([d["tag"] for d in points])
    return X, y, tag
