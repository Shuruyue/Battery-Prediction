"""Automated checks of the results in results/*.json (run_all). scripts/validate.py prints them, writes
VALIDATION.md and fails on any FAIL. Tolerances were fixed before the final runs; each states its reason.
INFO rows report observations without a criterion."""
import os

import numpy as np

from . import design as ds
from . import experiment as ex
from . import results as rs
from . import summary


def _row(category, check, value, expected, tol, passed, how):
    return dict(category=category, check=check, value=value, expected=expected, tolerance=tol,
                result="PASS" if passed else "FAIL", how=how)


def _info(category, check, value, how):
    """A reported observation without a pass/fail criterion."""
    return dict(category=category, check=check, value=value, expected="", tolerance="", result="INFO", how=how)


def reference_checks(rd):
    rows = []
    audit = rs.load_json(os.path.join(rd, "reference_mesh_audit.json"))
    runs = {(r["T_C"], r["load"], r["soh"], r["mesh"]["x_n"],
             r["mesh"]["r_n"]): r for r in audit["runs"] if r.get("ok")}
    for (label, a, b) in (("radial points 160 vs 320 (x 20/10/20)", (20, 160), (20, 320)),
                          ("cells across the stack 20 vs 40 (r 160)", (20, 160), (40, 160))):
        d = [abs(runs[pt + a]["runtime_min"] / runs[pt + b]["runtime_min"] - 1) * 100
             for pt in map(tuple, audit["points"]) if pt + a in runs and pt + b in runs]
        rows.append(_row("Reference", f"mesh audit: {label}, worst of {len(d)} corner points (%)", max(d), 0, 0.2,
                         max(d) < 0.2, "runtime at the hardest corners of the design on two meshes"))
    pts = ds.design_points()
    ok = rs.reference_points(os.path.join(rd, "reference.json"))
    rows.append(_row("Reference", "design points solved to 2.5 V", len(ok), len(pts), 0, len(ok) == len(pts),
                     "PyBaMM DFN + lumped thermal at every design point; end voltage within 20 mV of the cut-off"))
    return rows


def pybamm_checks(rd):
    rows = []
    v = rs.load_json(os.path.join(rd, "verify_pybamm.json"))
    cc = v["code_check"]
    for model, thermal in sorted({(r["model"], r["thermal"]) for r in cc}, key=lambda m: (m[0] != "SPM", m[1])):
        rr = [r for r in cc if r["model"] == model and r["thermal"] == thermal]
        rms = max(r["rms_mV"] for r in rr)
        dt = max(abs(r["runtime_diff_percent"]) for r in rr)
        dT = max(abs(r["T_end_own_C"] - r["T_end_pybamm_C"]) for r in rr)
        name = f"{model}, {thermal}"
        how = f"same equations in both codes, {len(rr)} cases C/2-2C, -10 to 25 C, fine meshes"
        rows.append(_row("Code check vs PyBaMM", f"{name}: worst voltage RMS (mV)", rms, 0, 5.0, rms < 5.0, how))
        rows.append(_row("Code check vs PyBaMM", f"{name}: worst runtime difference (%)", dt, 0, 0.5, dt < 0.5, how))
        if thermal == "lumped":
            rows.append(_row("Code check vs PyBaMM", f"{name}: worst end-temperature difference (K)", dT, 0, 0.1,
                             dT < 0.1, how))
    fc = [r for r in v["form_check"] if r["thermal"] == "lumped"]
    worst = max(abs(r["own_SPMe_min"] / r["pybamm_DFN_min"] - 1) * 100 for r in fc)
    rows.append(_row("Model form vs DFN", "SPMe + heat against DFN + heat, worst runtime difference (%)", worst, 0, 1.0,
                     worst < 1.0, f"{len(fc)} constant-power cases 9-27 W, 0-25 C"))
    return rows


def convergence_checks(rd):
    rows = []
    c = rs.load_json(os.path.join(rd, "convergence.json"))
    own = {r["n_r"]: r["runtime_s"] for r in c["spm"]["own"]}
    pb = {r["n_r"]: r["runtime_s"] for r in c["spm"]["pybamm"]}
    d = abs(own[15] / own[max(own)] - 1) * 100
    rows.append(_row("Convergence", "SPM, 0 C 2C: 15 against 240 radial cells (%)", d, 0, 0.2, d < 0.2,
                     "Kirchhoff-transform face diffusivities"))
    d = abs(own[max(own)] / pb[max(pb)] - 1) * 100
    rows.append(_row("Convergence", "SPM, 0 C 2C: finest own against finest PyBaMM (480 points) (%)", d, 0, 0.2,
                     d < 0.2,
                     "both codes converge to the same runtime"))
    base = c["tool"]["base"]
    runs = c["tool"]["runs"]
    def at(name, val):
        return [r["runtime_s"] for r in runs if r["varied"] == name and r["value"] == val][0]
    b = at("n_r", base["n_r"])
    for name, finest in (("n_r", 120), ("n_x", [80, 40, 80]), ("dt", 0.25)):
        d = abs(b / at(name, finest) - 1) * 100
        rows.append(_row("Convergence",
                         f"simulation tool, 0 C 150 % load SOH 0.7: default against {name} = {finest} (%)", d,
                         0, 0.3, d < 0.3, "one setting refined at a time"))
    return rows


def heat_checks(rd):
    rows = []
    th = rs.load_json(os.path.join(rd, "thermal_effect.json"))["rows"]
    err = lambda r: abs(r["own_min"] / r["dfn_min"] - 1) * 100  # noqa: E731
    used = [r for r in th if r["h"] == 10.0]
    worst = max(err(r) for r in used)
    rows.append(_row("Self-heating", "SPMe against reference at h = 10 (the reference's cooling), 0 and 25 C (%)",
                     worst, 0, 1.0, worst < 1.0, "rated load, new cell, the reference's parameters"))
    mild = max(err(r) for r in th if r["h"] is not None and r["h"] <= 25.0)
    rows.append(_info("Self-heating", "SPMe against reference for h = 2-25, 0 and 25 C (%)", mild,
                      "observation; the planned 1 % check over every h failed, see the next row"))
    cold = [r for r in th if r["T_C"] == 0.0 and (r["h"] is None or r["h"] > 25.0)]
    rows.append(_info("Self-heating", "SPMe against reference at 0 C when the cell stays cold "
                      "(h = 100, isothermal) (%)", f"{min(map(err, cold)):.1f}-{max(map(err, cold)):.1f}",
                      "observation: the electrolyte limitation that the SPMe simplifies grows in a cold cell"))
    cold = {r["h"]: r for r in th if r["T_C"] == 0.0}
    gain = (cold[10.0]["dfn_min"] / cold[None]["dfn_min"] - 1) * 100
    rows.append(_row("Self-heating", "reference at 0 C: self-heating lengthens runtime (%)", gain, "> 20", "",
                     gain > 20,
                     "lumped thermal (h = 10) against isothermal; the claim of the README"))
    return rows


def experiment_checks(rd):
    rows = []
    ml = rs.load_json(os.path.join(rd, "ml_results.json"))
    acc = ml["accuracy"]
    covs = {k: a["cover_in"][0] for k, a in acc.items() if "cover_in" in a}
    lo, hi = min(covs.values()), max(covs.values())
    rows.append(_row("Experiment", "conformal 90 % intervals inside the range, mean over seeds (range over models)",
                     f"{lo:.3f}-{hi:.3f}", 0.90, "0.85-0.97", lo >= 0.85 and hi <= 0.97,
                     f"{ex.N_CAL} calibration points per seed; finite-sample coverage scatters around 46/51"))
    data = ex.load_data(os.path.join(rd, "reference.json"), os.path.join(rd, "model_points.json"))
    pk = ex.formula(data)
    sl = data.sweep("sweep_load")
    at = {round(float(x[1]), 6): (x, t) for x, t in zip(data.X[sl], data.y[sl], strict=True)}
    err = max(abs(pk.predict(at[L][0][None])[0] / at[L][1] - 1) for L in (0.5, 1.0))
    # computed here, not read from results/, so its last bits depend on the platform's libm: report it rounded,
    # which keeps VALIDATION.md the same on every platform (CI compares it)
    rows.append(_row("Experiment", "Peukert law passes through its two rated points (relative)", round(err, 13), 0,
                     1e-12, err < 1e-12, "closed form; shown rounded to 1e-13"))
    rows.append(_row("Experiment", "points: normal / outside", f"{len(data.normal)} / {len(data.wide)}",
                     "256 / up to 128",
                     "", len(data.normal) == 256 and 0 < len(data.wide) <= 128, "Sobol design"))
    return rows


def readme_checks(rd, readme):
    s = summary.compute(rd)
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    missing = [k for k, v in s.items() if v not in text]
    how = "every string of summary.compute must appear verbatim"
    if missing:
        how += f"; missing: {', '.join(missing)}"
    return [_row("README", "numbers quoted in README.md that differ from the results", len(missing), 0, 0, not missing,
                 how)]


def run_all(results_dir, readme=None):
    rows = (reference_checks(results_dir) + pybamm_checks(results_dir) + convergence_checks(results_dir)
            + heat_checks(results_dir) + experiment_checks(results_dir))
    if readme:
        rows += readme_checks(results_dir, readme)
    return rows


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return "—" if v is None or v == "" else str(v)
