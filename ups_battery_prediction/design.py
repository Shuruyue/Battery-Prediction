"""Design of the experiment: the operating points at which the reference is run and every predictor is scored.

256 points in the normal operating range (Sobol), four one-at-a-time sweeps, and up to 128 points of a wider box
(Sobol) that fall outside the normal range. A point is (ambient temperature [C], load [fraction of the rated
18 W per cell], state of health).
"""
P_RATED = 18.0                       # W per cell: a 56-cell, 1 kW UPS at rated load
NORMAL = dict(T=(18.0, 28.0), load=(0.3, 0.8), soh=(0.9, 1.0))
WIDE = dict(T=(0.0, 45.0), load=(0.2, 1.5), soh=(0.7, 1.0))


def key(d):
    """Identity of a design point."""
    return d["tag"], round(d["T_C"], 6), round(d["load"], 6), round(d["soh"], 6)


def in_normal_range(T_C, load, soh):
    return ((NORMAL["T"][0] <= T_C <= NORMAL["T"][1]) and (NORMAL["load"][0] <= load <= NORMAL["load"][1])
            and soh >= NORMAL["soh"][0])


def design_points():
    from scipy.stats import qmc
    pts = []

    def add(tag, T_C, load, soh):
        pts.append(dict(tag=tag, T_C=float(T_C), load=float(load), soh=float(soh)))

    # widths written out (10, 0.5, 0.1 and 45, 1.3, 0.3): differences such as 0.8 - 0.3 are not exact in binary
    for u in qmc.Sobol(3, seed=7).random(256):
        add("normal", NORMAL["T"][0] + 10 * u[0], NORMAL["load"][0] + 0.5 * u[1], NORMAL["soh"][0] + 0.1 * u[2])
    for T in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
        add("sweep_T_0.6", T, 0.6, 1.0)
        add("sweep_T_1.0", T, 1.0, 1.0)
    for L in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        add("sweep_load", 25, L, 1.0)
    for h in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        add("sweep_soh", 25, 0.6, h)
    for u in qmc.Sobol(3, seed=11).random(128):
        T, L, h = WIDE["T"][0] + 45 * u[0], WIDE["load"][0] + 1.3 * u[1], WIDE["soh"][0] + 0.3 * u[2]
        if not in_normal_range(T, L, h):
            add("wide", T, L, h)
    return pts
