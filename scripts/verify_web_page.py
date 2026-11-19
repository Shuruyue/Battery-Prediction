"""Check web/battery.html against the results: the simulation grid and the reference points of its data block
equal results/model_grid.json and results/reference.json, and the page equals web/src/ built with its own data
block (nothing edited by hand).

Usage: python scripts/verify_web_page.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from battery_prediction import results as rs

PAGE = Path(_path.ROOT) / "web" / "battery.html"


def page_data():
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("window.RUNTIME_DATA = ") + len("window.RUNTIME_DATA = ")
    return json.loads(text[start:text.index(";</script>", start)]), text


def run():
    """{check: (number of items, number that differ)}."""
    data, text = page_data()
    report = {}
    g = rs.load_json(_path.result("model_grid.json"))
    sim = np.round(np.array(g["runtime_min"]).ravel(), 3)
    report["simulation grid"] = (sim.size, int(np.sum(np.abs(np.array(data["models"]["Simulation"]) - sim) > 1e-9)))
    pts = rs.reference_points(_path.result("reference.json"))
    ok = len(pts) == len(data["reference"]) and all(
        abs(a["t"] - round(b["runtime_min"], 3)) < 1e-9 and a["tag"] == b["tag"]
        for a, b in zip(data["reference"], pts, strict=True))
    report["reference points"] = (len(pts), 0 if ok else 1)
    spec = importlib.util.spec_from_file_location("build_web_page", Path(_path.ROOT) / "scripts" / "build_web_page.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    src = (build.SRC / "battery.src.html").read_text(encoding="utf-8")
    src = src.replace("/*__CHART__*/", (build.SRC / "chart.js").read_text(encoding="utf-8"))
    src = src.replace("/*__DATA__*/", "window.RUNTIME_DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    report["page built from web/src"] = (1, 0 if src == text else 1)
    return report


def main():
    report = run()
    for k, (n, bad) in report.items():
        print(f"{'ok  ' if bad == 0 else 'FAIL'} {k}: {n - bad} of {n} agree")
    sys.exit(1 if any(bad for _, bad in report.values()) else 0)


if __name__ == "__main__":
    main()
