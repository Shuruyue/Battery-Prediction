"""Draw the README figures from results/*.json into docs/figures/. About two minutes."""
import os

import _path  # noqa: F401

from ups_battery_prediction import plotting as pl
from ups_battery_prediction import results as rs

OUT = os.path.join(_path.ROOT, "docs", "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    r = _path.result
    ml = rs.load_json(r("ml_results.json"))
    jobs = {
        "error_vs_distance.png": lambda p: pl.error_vs_distance(ml, p),
        "self_heating.png": lambda p: pl.self_heating(rs.load_json(r("thermal_effect.json")), p),
        "model_forms.png": lambda p: pl.model_forms(_path.RESULTS, p),
        "sweeps.png": lambda p: pl.sweeps(ml, p),
        "parity.png": lambda p: pl.parity(ml, p),
        "coverage.png": lambda p: pl.coverage(ml, p),
        "runtime_map.png": lambda p: pl.runtime_map(rs.load_json(r("model_grid.json")), p),
        "convergence.png": lambda p: pl.convergence(rs.load_json(r("convergence.json")), p),
        "thermal_runaway.png": lambda p: pl.runaway(_path.RESULTS, p),
    }
    for name, draw in jobs.items():
        draw(os.path.join(OUT, name))
        print(name)


if __name__ == "__main__":
    main()
