"""Check web/ups-battery.html against the package.

1. The page's thermal-runaway model (the block between "// ---- physics: begin" and "// ---- physics: end") is
   run in Node.js and compared with ups_battery_prediction.thermal_runaway: oven tests at random chemistry, oven
   temperature and cooling (runaway or not, runaway time, peak temperature), the critical oven temperature of
   each chemistry and the number of cells in runaway along a row.
2. The data block: the simulation grid and the reference points equal results/model_grid.json and
   results/reference.json, and the page equals web/src/ built with its own data block (nothing edited by hand).

Usage: python scripts/verify_web_page.py [--cases 40]
Requires Node.js on the PATH.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from ups_battery_prediction import reference as ref
from ups_battery_prediction import thermal_runaway as tr

PAGE = Path(_path.ROOT) / "web" / "ups-battery.html"
CHEMS = {"LCO": tr.LCO, "NMC111": tr.NMC, "LFP": tr.LFP}

NODE_HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const page = fs.readFileSync(process.argv[1], "utf8");
const after = page.split("// ---- physics: begin")[1];
const block = after.slice(after.indexOf("\n")).split("// ---- physics: end")[0];
const ctx = {}; vm.createContext(ctx); vm.runInContext(block, ctx);
const TR = ctx.TR, input = JSON.parse(fs.readFileSync(0, "utf8")), out = { ovens: [], rows: [], crit: {} };
for (const c of input.ovens) {
  const s = TR.ovenTest(TR.CHEM[c.chem], c.T, { h: c.h, rtol: 1e-5 });
  out.ovens.push([s.runaway, s.tRunaway, s.peak]);
}
for (const c of input.rows) out.rows.push(TR.propagation(TR.CHEM[c.chem], { G: c.G }).nRun);
for (const c of input.crit) out.crit[c] = TR.criticalOven(TR.CHEM[c]);
process.stdout.write(JSON.stringify(out, (k, v) => (typeof v === "number" && !isFinite(v)) ? null : v));
"""


def cases(n, seed=0):
    rng = random.Random(seed)
    ovens = [dict(chem=rng.choice(list(CHEMS)), T=rng.uniform(410.0, 500.0), h=rng.uniform(5.0,
                                                                                           40.0)) for _ in range(n)]
    rows = [dict(chem=c, G=G) for c in CHEMS for G in (0.05, 0.2, 1.0)]
    return dict(ovens=ovens, rows=rows, crit=list(CHEMS))


def run_node(inp):
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is needed to run the page's JavaScript")
    res = subprocess.run([node, "-e", NODE_HARNESS, str(PAGE)], input=json.dumps(inp), capture_output=True,
                         text=True, encoding="utf-8", check=True)
    return json.loads(res.stdout)


def check_physics(n_cases):
    from dataclasses import replace
    inp = cases(n_cases)
    js = run_node(inp)
    report = {}
    bad = 0
    for c, (run_js, t_js, peak_js) in zip(inp["ovens"], js["ovens"], strict=True):
        t, y, s = tr.oven_test(CHEMS[c["chem"]], c["T"], cell=replace(tr.Cell(), h=c["h"]), rtol=1e-5)
        t_py = tr.crossing_time(t, y[:, 0], c["T"] + 100.0) if s["runaway"] else None
        ok = run_js == s["runaway"] and abs(peak_js - s["T_peak"]) < 0.5
        if s["runaway"]:
            ok = ok and abs(t_js / t_py - 1) < 2e-3
        bad += not ok
    report["oven tests"] = (len(inp["ovens"]), bad)
    bad = 0
    for c, n_js in zip(inp["rows"], js["rows"], strict=True):
        bad += n_js != tr.propagation(CHEMS[c["chem"]], n=6, G=c["G"], t_end=2400.0)[2]["n_runaway"]
    report["rows of six cells"] = (len(inp["rows"]), bad)
    crit = ref.load_json(_path.result("verify_runaway.json"))["oven"]
    bad = sum(abs(js["crit"][c] - 273.15 - crit[f"{c}_critical_C"]) > 0.3 for c in CHEMS)
    report["critical oven temperatures"] = (len(CHEMS), bad)
    return report


def page_data():
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("window.RUNTIME_DATA = ") + len("window.RUNTIME_DATA = ")
    return json.loads(text[start:text.index(";</script>", start)]), text


def check_data():
    import importlib.util
    data, text = page_data()
    report = {}
    g = ref.load_json(_path.result("model_grid.json"))
    sim = np.round(np.array(g["runtime_min"]).ravel(), 3)
    report["simulation grid"] = (sim.size, int(np.sum(np.abs(np.array(data["models"]["Simulation"]) - sim) > 1e-9)))
    pts = ref.reference_points(_path.result("reference.json"))
    ok = len(pts) == len(data["reference"]) and all(abs(a["t"] - round(b["runtime_min"],
                                                                       3)) < 1e-9 and a["tag"] == b["tag"]
                                                     for a, b in zip(data["reference"], pts, strict=True))
    report["reference points"] = (len(pts), 0 if ok else 1)
    spec = importlib.util.spec_from_file_location("build_web_page", Path(_path.ROOT) / "scripts" / "build_web_page.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    src = (build.SRC / "ups-battery.src.html").read_text(encoding="utf-8")
    for token, f in (("/*__CHART__*/", "chart.js"), ("/*__MODEL__*/", "runaway_model.js")):
        src = src.replace(token, (build.SRC / f).read_text(encoding="utf-8"))
    src = src.replace("/*__DATA__*/", "window.RUNTIME_DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    report["page built from web/src"] = (1, 0 if src == text else 1)
    return report


def run(n_cases=40):
    return {**check_physics(n_cases), **check_data()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    report = run(ap.parse_args().cases)
    for k, (n, bad) in report.items():
        print(f"{'ok  ' if bad == 0 else 'FAIL'} {k}: {n - bad} of {n} agree")
    sys.exit(1 if any(bad for _, bad in report.values()) else 0)


if __name__ == "__main__":
    main()
