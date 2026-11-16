"""Figures of the README, drawn from results/*.json.

Style of Materials-ML-Primer (matplotlib defaults, 10 pt, titled panels), plus three rules: one colour per kind
of tool in every figure and in the interactive page, names written next to the lines where possible, and
square axes for predicted-against-reference plots.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import reference as ref  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10, "savefig.bbox": "tight"})
COLORS = {"reference": "#141412", "simulation": "#2a78d6", "formula": "#eb6834", "ml": "#d03b3b",
          "physics": "#1baf7a", "hybrid": "#6250d6", "band": "#2a78d6"}
TOOLS = [  # (key in ml_results, label, colour, line style)
    ("Formula (Peukert)", "formula", COLORS["formula"], "-"),
    ("Simulation (own model)", "simulation", COLORS["simulation"], "-"),
    ("Gradient boosting", "gradient boosting", COLORS["ml"], "-"),
    ("Neural network", "neural network", COLORS["ml"], "--"),
    ("Gaussian process", "Gaussian process", COLORS["ml"], ":"),
    ("Physics features + linear fit", "physics features", COLORS["physics"], "-"),
    ("Simulation + data correction", "simulation + data", COLORS["hybrid"], "-"),
]


def _label_end(ax, x, y, text, color, dy=0.0, log=False):
    """Write a series name just right of its last point."""
    yy = y * 10 ** dy if log else y + dy
    ax.annotate(text, (x, yy), xytext=(4, 0), textcoords="offset points", va="center", color=color, fontsize=8.5)


def _spread(values, gap, log=False):
    """Positions of end labels, pushed apart so that they do not overlap (in log10 units when log)."""
    v = np.log10(values) if log else np.asarray(values, float)
    order = np.argsort(v)
    pos = v.copy()
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        pos[i] = max(pos[i], pos[j] + gap)
    pos -= (pos - v).mean() * 0.5
    return pos - v


def error_vs_distance(ml, path):
    """Hook figure: error of each tool against the distance of the test point from the training range."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bins = [b["bin"] for b in ml["by_distance"]["Formula (Peukert)"]]
    keep = [i for i, b in enumerate(ml["by_distance"]["Formula (Peukert)"]) if b["n"] > 0]
    xs = np.array([0.5 * (bins[i][0] + min(bins[i][1], 6.0)) for i in keep])
    ends, rows = [], []
    for key, label, color, ls in TOOLS:
        ys = np.array([ml["by_distance"][key][i]["mape"] for i in keep])
        ax.plot(xs, ys, ls, color=color, marker="o", ms=4, lw=1.8)
        ends.append(ys[-1])
        rows.append((label, color))
    shift = _spread(ends, 0.09, log=True)
    for (label, color), y, dy in zip(rows, ends, shift, strict=True):
        _label_end(ax, xs[-1], y, label, color, dy, log=True)
    ax.set_yscale("log")
    ax.set_xlim(0, xs[-1] * 1.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{bins[i][0]:g}–{bins[i][1]:g}" if bins[i][1] < 50 else f"> {bins[i][0]:g}" for i in keep])
    ns = [ml["by_distance"]["Formula (Peukert)"][i]["n"] for i in keep]
    for x, n in zip(xs, ns, strict=True):
        ax.annotate(f"n = {n}", (x, 0), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color="0.4")
    ax.set_xlabel("distance of the test point from the training range (box widths)")
    ax.set_ylabel("mean absolute error of runtime (%)")
    ax.set_title("Outside the training range, error grows fastest for the learned models")
    fig.savefig(path)
    plt.close(fig)


def self_heating(th, path):
    """Runtime at rated load against the cooling coefficient, 0 and 25 C, reference and simulation."""
    rows = th["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    for T_C, mk in ((0.0, "o"), (25.0, "s")):
        rr = sorted([r for r in rows if r["T_C"] == T_C and r["h"] is not None], key=lambda r: r["h"])
        iso = [r for r in rows if r["T_C"] == T_C and r["h"] is None][0]
        h = [r["h"] for r in rr]
        ax.plot(h, [r["dfn_min"] for r in rr], "-", marker=mk, color=COLORS["reference"], ms=5)
        ax.plot(h, [r["own_min"] for r in rr], "--", marker=mk, color=COLORS["simulation"], ms=4, mfc="none")
        ax.axhline(iso["dfn_min"], color=COLORS["reference"], lw=0.8, ls=":")
        below = T_C > 10.0      # the 25 C label goes under its line, the 0 C label above
        ax.annotate(f"{T_C:.0f} °C, cell held at room temperature", (h[-1], iso["dfn_min"]),
                    xytext=(0, -3 if below else 3), textcoords="offset points", ha="right",
                    va="top" if below else "bottom", fontsize=8, color="0.3")
        ax.annotate(f"room {T_C:.0f} °C", (h[-1], rr[-1]["dfn_min"]), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5)
    ax.axvline(10.0, color="0.6", lw=0.8, zorder=0)
    ax.annotate("h of the reference", (10.0, 0.97), xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=8, color="0.4", va="top")
    ax.set_xscale("log")
    ax.set_xlim(1.6, 260)
    ax.set_xlabel("heat-transfer coefficient h (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel("runtime at rated load (min)")
    ax.set_title("A cell that heats itself outlasts an isothermal one")
    ax.plot([], [], "-", color=COLORS["reference"], label="reference (DFN)")
    ax.plot([], [], "--", color=COLORS["simulation"], label="simulation (SPMe)")
    ax.legend(loc="center left", fontsize=8)
    ax = axes[1]
    for T_C, mk in ((0.0, "o"), (25.0, "s")):
        rr = sorted([r for r in rows if r["T_C"] == T_C and r["h"] is not None], key=lambda r: r["h"])
        ax.plot([r["h"] for r in rr], [r["dfn_T_peak_C"] for r in rr], "-", marker=mk, color=COLORS["reference"], ms=5)
        ax.annotate(f"start {T_C:.0f} °C", (rr[-1]["h"], rr[-1]["dfn_T_peak_C"]), xytext=(-4, 6),
                    textcoords="offset points", ha="right", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("heat-transfer coefficient h (W m$^{-2}$ K$^{-1}$)")
    ax.set_ylabel("cell temperature at the end (°C)")
    ax.set_title("Temperature at the end of the discharge")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_forms(results_dir, path):
    """Error of three forms of the hand-written model against the reference, along the temperature sweep."""
    pts = {ref.key(d): d for d in ref.reference_points(os.path.join(results_dir, "reference.json"))}
    own = {ref.key(d): d for d in ref.load_json(os.path.join(results_dir, "model_points.json"))}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    forms = [("SPM_isothermal", "single particle, isothermal", COLORS["simulation"], ":"),
             ("SPMe_isothermal", "+ electrolyte, isothermal", COLORS["simulation"], "--"),
             ("SPMe_lumped", "+ electrolyte + self-heating", COLORS["simulation"], "-")]
    for ax, tag, title in zip(axes, ("sweep_T_0.6", "sweep_T_1.0"), ("60 % of rated power", "rated power"),
                              strict=True):
        ks = sorted([k for k in pts if k[0] == tag], key=lambda k: k[1])
        T = [k[1] for k in ks]
        at5 = []
        for col, _label, color, ls in forms:
            e = [100 * (own[k][col] / pts[k]["runtime_min"] - 1) for k in ks]
            ax.plot(T, e, ls, color=color, lw=1.8, marker="o", ms=3)
            at5.append(e[1])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xlabel("room temperature (°C)")
        ax.set_title(f"{title}, new cell")
        if ax is axes[0]:
            ax.set_ylabel("runtime error against the reference (%)")
        else:   # the three curves are far apart here: name them next to the 5 C point
            for (_col, label, color, _ls), y in zip(forms, at5, strict=True):
                ax.annotate(label, (T[1], y), xytext=(8, 6 if abs(y) < 1 else 0), textcoords="offset points",
                            fontsize=8.5, color=color, va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sweeps(ml, path):
    """Reference and predictions along the temperature and state-of-health sweeps (seed 0)."""
    s0 = ml["seed0"]
    X, y, tag = np.array(s0["X"]), np.array(s0["y"]), np.array(s0["tag"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, t, col, xlabel, band in ((axes[0], "sweep_T_1.0", 0, "room temperature (°C)", ref.NORMAL["T"]),
                                     (axes[1], "sweep_soh", 2, "state of health", (ref.NORMAL["soh"][0], 1.0))):
        idx = np.where(tag == t)[0]
        idx = idx[np.argsort(X[idx, col])]
        x = X[idx, col]
        ax.axvspan(*band, color=COLORS["band"], alpha=0.10, lw=0)
        ax.plot(x, y[idx], "o", color=COLORS["reference"], ms=5, zorder=5, label="reference")
        for key, label, color, ls in TOOLS:
            ax.plot(x, np.array(s0["preds"][key])[idx], ls, color=color, lw=1.6, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("runtime (min)")
    axes[0].set_title("Rated load (outside the training range), new cell")
    axes[1].set_title("60 % load, 25 °C")
    axes[1].legend(fontsize=7.5, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parity(ml, path):
    """Predicted against reference runtime, inside and outside the range (seed 0), square axes."""
    s0 = ml["seed0"]
    y, tag = np.array(s0["y"]), np.array(s0["tag"])
    test = np.array(s0["test_idx"])
    wide = np.where(tag == "wide")[0]
    show = [("Simulation (own model)", "simulation"), ("Gaussian process", "Gaussian process"),
            ("Physics features + linear fit", "physics features")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    hi = 1.1 * y.max()
    for ax, (key, label) in zip(axes, show, strict=True):
        p = np.array(s0["preds"][key])
        color = {t[0]: t[2] for t in TOOLS}[key]
        ax.plot([1, hi], [1, hi], color="0.6", lw=0.8)
        ax.scatter(y[wide], p[wide], s=12, facecolors="none", edgecolors=color, lw=0.8, label="outside")
        ax.scatter(y[test], p[test], s=12, color=color, label="inside")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(20, hi)
        ax.set_ylim(20, hi)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("reference runtime (min)")
    axes[0].set_ylabel("predicted runtime (min)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def coverage(ml, path):
    """Coverage of the 90 % conformal intervals inside and outside the training range."""
    acc = ml["accuracy"]
    keys = [k for k, *_ in TOOLS if "cover_in" in acc[k]] + ["Linear regression", "Polynomial (degree 3)",
                                                             "Random forest"]
    keys = list(dict.fromkeys(keys))
    names = {t[0]: t[1] for t in TOOLS}
    names.update({"Linear regression": "linear regression", "Polynomial (degree 3)": "polynomial",
                  "Random forest": "random forest"})
    colors = {t[0]: t[2] for t in TOOLS}
    keys.sort(key=lambda k: acc[k]["cover_wide"][0])
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    yy = np.arange(len(keys))
    ax.barh(yy - 0.18, [100 * acc[k]["cover_in"][0] for k in keys], 0.34, color="0.75", label="inside the range")
    ax.barh(yy + 0.18, [100 * acc[k]["cover_wide"][0] for k in keys], 0.34,
            color=[colors.get(k, COLORS["ml"]) for k in keys], label="outside the range")
    ax.axvline(90, color="0.3", lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([names[k] for k in keys])
    ax.set_xlabel("share of test points inside the 90 % interval (%)")
    ax.set_xlim(0, 100)
    ax.set_title("A 90 % interval calibrated inside the range")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(path)
    plt.close(fig)


def runtime_map(grid, path):
    """Simulated runtime over temperature and load (new cell), training range outlined."""
    T, L = np.array(grid["T_C"]), np.array(grid["load"])
    G = np.array(grid["runtime_min"])[:, :, 0]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    levels = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
    cs = ax.contourf(T, 100 * L, G.T, levels=levels, cmap="viridis", extend="both")
    cl = ax.contour(T, 100 * L, G.T, levels=levels, colors="white", linewidths=0.6)
    ax.clabel(cl, fmt="%g", fontsize=7.5)
    lo, hi = ref.NORMAL["T"], ref.NORMAL["load"]
    ax.add_patch(plt.Rectangle((lo[0], 100 * hi[0]), lo[1] - lo[0], 100 * (hi[1] - hi[0]), fill=False, ec="white",
                               lw=1.6, ls="--"))
    ax.annotate("training range", (lo[1], 100 * hi[1]), xytext=(3, 3), textcoords="offset points", color="white",
                fontsize=8.5)
    fig.colorbar(cs, ax=ax, label="runtime (min)")
    ax.set_xlabel("room temperature (°C)")
    ax.set_ylabel("load (% of rated power)")
    ax.set_title("Simulated runtime, new cell")
    fig.savefig(path)
    plt.close(fig)


def convergence(conv, path):
    """Radial convergence of the single-particle model, ours against PyBaMM's (0 C, 2C)."""
    sp = conv["spm"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ref_t = sp["pybamm"][-1]["runtime_s"]
    for rows, label, color, mk in ((sp["own"], "this code (Kirchhoff faces)", COLORS["simulation"], "o"),
                                   (sp["pybamm"], "PyBaMM SPM", COLORS["reference"], "s")):
        n = [r["n_r"] for r in rows]
        e = [max(abs(r["runtime_s"] / ref_t - 1) * 100, 1e-3) for r in rows]
        ax.plot(n, e, "-", marker=mk, color=color, label=label)
    ax.axvline(20, color="0.6", lw=0.8)
    ax.annotate("PyBaMM default", (20, 0.03), xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points",
                rotation=90, fontsize=8, color="0.4", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("radial points per particle")
    ax.set_ylabel("runtime difference from the finest PyBaMM run (%)")
    ax.set_title("0 °C, 2C discharge, single-particle model")
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def runaway(results_dir, path):
    """Oven tests of the three chemistries and propagation along a row."""
    from . import thermal_runaway as tr
    run = ref.load_json(os.path.join(results_dir, "verify_runaway.json"))
    sc = ref.load_json(os.path.join(results_dir, "runaway_scenarios.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    shades = {"LCO": "0.1", "NMC111": "0.4", "LFP": "0.62"}
    for ch in (tr.LCO, tr.NMC, tr.LFP):
        crit = run["oven"][f"{ch.name}_critical_C"]
        for dT, ls in ((+2.0, "-"), (-2.0, ":")):
            t, y, _ = tr.oven_test(ch, crit + dT + 273.15, t_end=10800.0)
            T = y[:, 0] - 273.15
            ax.plot(t / 60, T, ls, color=shades[ch.name], lw=1.4)
            if dT > 0:      # name the chemistry at its runaway peak
                k = int(np.argmax(T))
                late = t[k] / 60 > 100.0      # keep the label inside the axes
                ax.annotate(f"{ch.name}\nonset {crit:.0f} °C", (t[k] / 60, T[k]), xytext=(-5 if late else 5, 3),
                            textcoords="offset points", fontsize=8.5, color=shades[ch.name], va="bottom",
                            ha="right" if late else "left")
    ax.set_xlabel("time in the oven (min)")
    ax.set_ylabel("cell temperature (°C)")
    ax.set_ylim(0, 600)
    ax.set_title("Oven 2 K above (solid) and below (dotted) onset")
    ax = axes[1]
    for name in ("LCO", "NMC111", "LFP"):
        rr = [r for r in sc["propagation"] if r["chem"] == name]
        ax.plot([r["G"] for r in rr], [r["n_runaway"] for r in rr], "-o", ms=4, color=shades[name], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("thermal conductance between neighbours (W/K)")
    ax.set_ylabel("cells in runaway (of 6)")
    ax.set_title("A row of six, the first cell at 250 °C")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
