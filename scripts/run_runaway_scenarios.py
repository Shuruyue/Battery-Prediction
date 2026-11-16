"""Two abuse scenarios of the thermal-runaway model, for the three chemistries:

* propagation: a row of six cells with the first one at 250 C, for neighbour conductances G from 0.02 to 3 W/K;
* internal short: one cell with a constant internal heat source, from the critical power up to ten times it,
  with the times at which the cell passes 60 C (a typical alarm), 80 C and 300 C.

Writes results/runaway_scenarios.json. About ten minutes on one core.
"""
import _path  # noqa: F401
import numpy as np

from ups_battery_prediction import results as rs
from ups_battery_prediction import thermal_runaway as tr

G_VALUES = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
SHORT_MULTIPLES = (1.1, 1.5, 2.0, 3.0, 5.0, 10.0)


def main():
    out = {"propagation": [], "short": []}
    c = tr.Cell()
    A_face = c.height * 2 * c.radius
    for ch in (tr.LCO, tr.NMC, tr.LFP):
        for G in G_VALUES:
            _, T, s = tr.propagation(ch, n=6, G=G, t_end=2400.0)
            rec = dict(chem=ch.name, G=G, gap_mm_air=1e3 * 0.03 * A_face / G, n_runaway=s["n_runaway"],
                       t_runaway_s=[None if np.isnan(x) else x for x in s["t_runaway_s"]],
                       T_peak_C=[float(v - 273.15) for v in T.max(axis=0)])
            out["propagation"].append(rec)
            print(rec["chem"], "G", G, "cells in runaway", rec["n_runaway"], flush=True)
    for ch in (tr.LCO, tr.NMC, tr.LFP):
        pc = tr.critical_short_power(ch)
        for mult in SHORT_MULTIPLES:
            t, y, s = tr.internal_short(ch, pc * mult)
            T = y[:, 0]
            rec = dict(chem=ch.name, P_crit_W=pc, P_W=pc * mult, multiple=mult, runaway=s["runaway"],
                       T_peak_C=s["T_peak"] - 273.15, t_60C_s=tr.crossing_time(t, T, 333.15),
                       t_80C_s=tr.crossing_time(t, T, 353.15), t_300C_s=tr.crossing_time(t, T, 573.15))
            out["short"].append(rec)
            print(rec["chem"], "short", round(rec["P_W"], 2), "W", "runaway" if rec["runaway"] else "no runaway",
                  flush=True)
    rs.save_json(out, _path.result("runaway_scenarios.json"), indent=1)


if __name__ == "__main__":
    main()
