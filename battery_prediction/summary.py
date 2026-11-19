"""Main results, computed from results/*.json. README.md quotes these numbers; scripts/validate.py checks that
every quoted string is still what the results give."""
import os

from . import ageing, params
from . import design as ds
from . import experiment as ex
from . import results as rs

N_CELLS = 56                          # example pack: 56 cells in series at 18 W each, about 1 kW
E_CELL_WH = 5.0 * 3.63                # nominal energy per cell (5 A h x 3.63 V)
TOOL_ROWS = [(ex.FORMULA, "Peukert formula"),
             (ex.SIM, "Simulation (SPMe + heat, parameters 30 % off, no fitting)"),
             ("Gradient boosting", "Gradient boosting"),
             ("Neural network", "Neural network"),
             ("Gaussian process", "Gaussian process"),
             (ex.PHYSICS, "Linear model on physics features"),
             (ex.HYBRID, "Simulation + data correction")]


def _pct(v):
    """Percent with one decimal, or two below 0.1 % (so that a small error does not print as 0.0 %)."""
    return f"{v:.{1 if abs(v) >= 0.1 else 2}f} %"


def _span(values, fmt):
    return f"{fmt(min(values))}–{fmt(max(values))} %"


def compute(results_dir):
    def load(name):
        return rs.load_json(os.path.join(results_dir, name))
    ml = load("ml_results.json")
    th = {(r["T_C"], r["h"]): r for r in load("thermal_effect.json")["rows"]}
    pert = {k: v["accuracy"] for k, v in load("perturbation.json")["cases"].items()}
    acc = ml["accuracy"]
    out = {}
    out["e_over_p_min"] = f"{E_CELL_WH / ds.P_RATED * 60:.1f} min"
    out["energy_kwh"] = f"{N_CELLS * E_CELL_WH / 1000:.2f} kWh"
    for T_C, name in ((25.0, "warm"), (0.0, "cold")):
        out[f"ref_{name}_min"] = f"{th[(T_C, 10.0)]['dfn_min']:.1f} min"
        out[f"ref_{name}_iso_min"] = f"{th[(T_C, None)]['dfn_min']:.1f} min"
        out[f"ref_{name}_end_C"] = f"{th[(T_C, 10.0)]['dfn_T_peak_C']:.0f} °C"
    for key, (T_C, h) in (("cold_h2", (0.0, 2.0)), ("cold_h100", (0.0, 100.0))):
        out[f"ref_{key}_min"] = f"{th[(T_C, h)]['dfn_min']:.1f} min"
    out["accuracy_table"] = "\n".join(f"| {label} | {_pct(acc[k]['mape_in'][0])} | {_pct(acc[k]['mape_wide'][0])} |"
                                      for k, label in TOOL_ROWS)
    out["ml_in"] = _span([acc[k]["mape_in"][0] for k in ex.ML], lambda v: f"{v:.1f}")
    out["ml_out"] = _span([acc[k]["mape_wide"][0] for k in ex.ML], lambda v: f"{v:.0f}")
    out["formula_out"] = _pct(acc[ex.FORMULA]["mape_wide"][0])
    out["sim_out"] = _pct(acc[ex.SIM]["mape_wide"][0])
    out["cover_ml_in"] = _span([100 * acc[k]["cover_in"][0] for k in ex.ML], lambda v: f"{v:.0f}")
    out["cover_ml_out"] = _span([100 * acc[k]["cover_wide"][0] for k in ex.ML], lambda v: f"{v:.0f}")
    out["cover_physics_out"] = f"{100 * acc[ex.PHYSICS]['cover_wide'][0]:.0f} %"
    out["cover_hybrid_out"] = f"{100 * acc[ex.HYBRID]['cover_wide'][0]:.0f} %"
    out["exact_out"] = _pct(pert[ex.EXACT_CASE][ex.SIM]["mape_wide"][0])
    one = [v[ex.SIM]["mape_wide"][0] for k, v in pert.items() if k not in (ex.EXACT_CASE, ex.TOOL_CASE)]
    out["one_group_out"] = _span(one, lambda v: f"{v:.1f}")
    out["hybrid_worst_out"] = _pct(max(v[ex.HYBRID]["mape_wide"][0] for v in pert.values()))
    out["hybrid_cover_out"] = _span([100 * v[ex.HYBRID]["cover_wide"][0] for v in pert.values()], lambda v: f"{v:.0f}")
    out["n_wide"] = str(ml["n_wide"])
    out["pybamm_version"] = "PyBaMM " + load("verify_pybamm.json")["pybamm_version"]
    old = ageing.cell_state(0.7)
    out["sei_thickness_07"] = f"{old.r_film / params.SEI_RESISTIVITY * 1e6:.2f} µm"
    out["film_resistance_07"] = f"{old.r_contact * 1e3:.0f} mΩ"
    return out


def main_lines(results_dir):
    s = compute(results_dir)
    return [f"Example pack: {N_CELLS} cells, {s['energy_kwh']}, about 1 kW; energy / power = {s['e_over_p_min']}",
            f"Reference at rated load: 25 °C {s['ref_warm_min']} (isothermal {s['ref_warm_iso_min']}), "
            f"0 °C {s['ref_cold_min']} (isothermal {s['ref_cold_iso_min']}, cell ends at {s['ref_cold_end_C']})",
            "Mean absolute percentage error, inside / outside the training range:", s["accuracy_table"],
            f"90 % conformal intervals outside: learned models {s['cover_ml_out']}, "
            f"physics features {s['cover_physics_out']}, simulation + data {s['cover_hybrid_out']}",
            f"Simulation outside: exact parameters {s['exact_out']}, one group wrong {s['one_group_out']}; "
            f"corrected by data at most {s['hybrid_worst_out']} (coverage {s['hybrid_cover_out']})"]
