"""Draw the README figures from results/*.json into docs/figures/. Under a minute."""
import os

import _path  # noqa: F401

from battery_prediction import plotting as pl
from battery_prediction import results as rs

OUT = os.path.join(_path.ROOT, "docs", "figures")


def main():
    os.makedirs(OUT, exist_ok=True)
    ml = rs.load_json(_path.result("ml_results.json"))
    jobs = {
        "error_vs_distance.png": lambda p: pl.error_vs_distance(ml, p),
        "coverage.png": lambda p: pl.coverage(ml, p),
        "sweeps.png": lambda p: pl.sweeps(ml, p),
        "self_heating.png": lambda p: pl.self_heating(rs.load_json(_path.result("thermal_effect.json")), p),
    }
    for name, draw in jobs.items():
        draw(os.path.join(OUT, name))
        print(name)


if __name__ == "__main__":
    main()
