"""Main results, computed from results/*.json. README.md quotes these numbers; scripts/validate.py checks that
every quoted string is still what the results give."""
import os

from . import reference as ref

N_CELLS = 56                          # example UPS: 56 cells in series at 18 W each, about 1 kW
E_CELL_WH = 5.0 * 3.63                # nominal energy per cell (5 A h x 3.63 V)
TOOL_ROWS = [("Formula (Peukert)", "Peukert formula"),
             ("Simulation (own model)", "Simulation (SPMe + heat, no fitting)"),
             ("Gradient boosting", "Gradient boosting"),
             ("Neural network", "Neural network"),
             ("Gaussian process", "Gaussian process"),
             ("Physics features + linear fit", "Linear model on physics features"),
             ("Simulation + data correction", "Simulation + data correction")]
PURE_ML = ("Linear regression", "Polynomial (degree 3)", "Random forest", "Gradient boosting", "Neural network",
           "Gaussian process")
HEADLINE_ML = ("Gradient boosting", "Neural network", "Gaussian process")
ALL_WORSE = "all worse (D, j0 x0.7, R x1.3)"


def _pct(v):
    """Percent with one decimal, or two below 0.1 % (so that a small error does not print as 0.0 %)."""
    return f"{v:.{1 if abs(v) >= 0.1 else 2}f} %"


def compute(results_dir):
    def load(name):
        return ref.load_json(os.path.join(results_dir, name))
    ml = load("ml_results.json")
    th = {(r["T_C"], r["h"]): r for r in load("thermal_effect.json")["rows"]}
    pert = load("perturbation.json")["cases"]
    run = load("verify_runaway.json")
    acc = ml["accuracy"]
    pts = ref.reference_points(os.path.join(results_dir, "reference.json"))
    hot = max(pts, key=lambda d: d["T_peak_C"])
    out = {}
    out["e_over_p_min"] = f"{E_CELL_WH / ref.P_RATED * 60:.1f} min"
    out["energy_kwh"] = f"{N_CELLS * E_CELL_WH / 1000:.2f} kWh"
    for T_C, name in ((25.0, "warm"), (0.0, "cold")):
        out[f"ref_{name}_min"] = f"{th[(T_C, 10.0)]['dfn_min']:.1f} min"
        out[f"ref_{name}_iso_min"] = f"{th[(T_C, None)]['dfn_min']:.1f} min"
        out[f"ref_{name}_end_C"] = f"{th[(T_C, 10.0)]['dfn_T_peak_C']:.0f} °C"
    for key, (T_C, h) in (("cold_h2", (0.0, 2.0)), ("cold_h100", (0.0, 100.0))):
        out[f"ref_{key}_min"] = f"{th[(T_C, h)]['dfn_min']:.1f} min"
    rows = []
    for key, label in TOOL_ROWS:
        rows.append(f"| {label} | {_pct(acc[key]['mape_in'][0])} | {_pct(acc[key]['mape_wide'][0])} |")
    out["accuracy_table"] = "\n".join(rows)
    cov = [acc[k]["cover_wide"][0] for k in PURE_ML]
    out["cover_ml_out"] = f"{100 * min(cov):.0f}–{100 * max(cov):.0f} %"
    cov_in = [acc[k]["cover_in"][0] for k in PURE_ML]
    out["cover_ml_in"] = f"{100 * min(cov_in):.0f}–{100 * max(cov_in):.0f} %"
    out["cover_physics_out"] = f"{100 * acc['Physics features + linear fit']['cover_wide'][0]:.0f} %"
    out["cover_hybrid_out"] = f"{100 * acc['Simulation + data correction']['cover_wide'][0]:.0f} %"
    head = [acc[k] for k in HEADLINE_ML]
    out["ml_in"] = f"{min(a['mape_in'][0] for a in head):.1f}–{max(a['mape_in'][0] for a in head):.1f} %"
    out["ml_out"] = f"{min(a['mape_wide'][0] for a in head):.0f}–{max(a['mape_wide'][0] for a in head):.0f} %"
    out["formula_out"] = _pct(acc["Formula (Peukert)"]["mape_wide"][0])
    one = [v["outside"] for k, v in pert.items() if k not in ("exact parameters", ALL_WORSE)]
    out["perturbed_out"] = f"{min(one):.1f}–{max(one):.1f} %"
    out["all_worse_out"] = _pct(pert[ALL_WORSE]["outside"])
    out["exact_out"] = _pct(pert["exact parameters"]["outside"])
    cold = [abs(r["own_min"] / r["dfn_min"] - 1) * 100 for (T_C, h), r in th.items()
            if T_C == 0.0 and (h is None or h > 25.0)]
    out["sim_cold_kept_cold"] = f"{min(cold):.1f}–{max(cold):.1f} %"
    forms = ml["model_forms"]
    for col in ("SPMe_lumped", "SPMe_isothermal", "SPM_isothermal"):
        out[f"form_{col}_out"] = _pct(forms[col]["mape_wide"])
    out["hot_point"] = f"{hot['T_C']:.0f} °C, {100 * hot['load']:.0f} % load, state of health {hot['soh']:.2f}"
    out["hot_peak_C"] = f"{hot['T_peak_C']:.0f} °C"
    for c in ("LCO", "NMC111", "LFP"):
        out[f"crit_{c}"] = f"{run['oven'][f'{c}_critical_C']:.0f} °C"
    out["n_wide"] = str(ml["n_wide"])
    return out


def main_lines(results_dir):
    s = compute(results_dir)
    lines = [f"Example UPS: {N_CELLS} cells, {s['energy_kwh']}, about 1 kW; energy / power = {s['e_over_p_min']}",
             f"Reference at rated load: 25 °C {s['ref_warm_min']} (isothermal {s['ref_warm_iso_min']}), "
             f"0 °C {s['ref_cold_min']} (isothermal {s['ref_cold_iso_min']}, cell ends at {s['ref_cold_end_C']})",
             "Mean absolute percentage error, inside / outside the training range:", s["accuracy_table"],
             f"90 % conformal intervals outside: pure ML {s['cover_ml_out']}, "
             f"physics features {s['cover_physics_out']}, "
             f"simulation + data {s['cover_hybrid_out']}",
             f"Simulation outside with one parameter group wrong: {s['perturbed_out']}, three groups "
             f"{s['all_worse_out']} (exact {s['exact_out']})",
             f"Hottest reference point: {s['hot_point']}, cell reaches {s['hot_peak_C']}",
             f"Critical oven temperature: LCO {s['crit_LCO']}, NMC111 {s['crit_NMC111']}, LFP {s['crit_LFP']}"]
    return lines
