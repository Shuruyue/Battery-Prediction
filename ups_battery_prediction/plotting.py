"""Figures of the README, drawn from results/*.json.

Journal style of Urea-Nitrate-Cell: 14 cm wide, 300 dpi, white background, serif text (Liberation Serif, or the
metrically identical Times New Roman) with STIX math, ticks inward on all four sides, framed legends; tick and
legend text 9.5 pt, axis titles 10.5 pt, which are the printed sizes when a figure is placed 14 cm wide. A panel
carries a short descriptive title with (a), (b), ...; what the figure shows is stated in its README caption.
One colour per kind of tool in every figure and in the interactive page; series are named next to their lines
where there is room.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

from . import design as ds  # noqa: E402
from . import results as rs  # noqa: E402

W_CM = 14.0
FS, FS_LAB = 9.5, 10.5
INK, GRAY, LIGHT = "#000000", "#5a5a5a", "#bdbdbd"
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
SERIF = [f for f in ("Liberation Serif", "Times New Roman") if f in _INSTALLED] + ["DejaVu Serif"]
plt.rcParams.update({
    "font.family": SERIF, "font.size": FS,
    "mathtext.fontset": "stix", "axes.labelsize": FS_LAB, "axes.titlesize": FS, "xtick.labelsize": FS,
    "ytick.labelsize": FS, "legend.fontsize": FS - 0.5, "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": INK, "legend.framealpha": 1,
    "legend.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.unicode_minus": True, "axes.edgecolor": INK,
})
COLORS = {"reference": "#141412", "simulation": "#2a78d6", "formula": "#eb6834", "ml": "#d03b3b",
          "physics": "#1baf7a", "hybrid": "#6250d6", "band": "#2a78d6"}
TOOLS = [  # (key in ml_results, label, colour, line style)
    ("Formula (Peukert)", "Formula", COLORS["formula"], "-"),
    ("Simulation (own model)", "Simulation", COLORS["simulation"], "-"),
    ("Gradient boosting", "Gradient boosting", COLORS["ml"], "-"),
    ("Neural network", "Neural network", COLORS["ml"], "--"),
    ("Gaussian process", "Gaussian process", COLORS["ml"], ":"),
    ("Physics features + linear fit", "Physics features", COLORS["physics"], "-"),
    ("Simulation + data correction", "Simulation + data", COLORS["hybrid"], "-"),
]
H_UNIT = r"(W m$^{-2}$ K$^{-1}$)"


def newfig(aspect, ncols=1, **kw):
    """A figure W_CM wide whose height is `aspect` times its width."""
    w = W_CM / 2.54
    return plt.subplots(1, ncols, figsize=(w, w * aspect), layout="constrained", **kw)


def save(fig, path):
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _label_end(ax, x, y, text, color, dy=0.0, log=False):
    """Write a series name just right of its last point."""
    yy = y * 10 ** dy if log else y + dy
    ax.annotate(text, (x, yy), xytext=(4, 0), textcoords="offset points", va="center", color=color, fontsize=FS - 0.5)


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
    fig, ax = newfig(0.62)
    by = ml["by_distance"]["Formula (Peukert)"]
    keep = [i for i, b in enumerate(by) if b["n"] > 0]
    bins = [by[i]["bin"] for i in keep]
    xs = np.array([0.5 * (lo + min(hi, 6.0)) for lo, hi in bins])
    ends, rows = [], []
    for key, label, color, ls in TOOLS:
        ys = np.array([ml["by_distance"][key][i]["mape"] for i in keep])
        ax.plot(xs, ys, ls, color=color, marker="o", ms=3.5)
        ends.append(ys[-1])
        rows.append((label, color))
    for (label, color), y, dy in zip(rows, ends, _spread(ends, 0.27, log=True), strict=True):
        _label_end(ax, xs[-1], y, label, color, dy, log=True)
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.8)              # room for the top label
    ax.set_xlim(0, xs[-1] * 1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([(f"{lo:g}–{hi:g}" if hi < 50 else f"> {lo:g}") + f"\n($n$ = {by[i]['n']})"
                        for (lo, hi), i in zip(bins, keep, strict=True)])
    ax.minorticks_off()
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlabel("Distance of the test point from the training range (box widths)")
    ax.set_ylabel("Mean absolute error of runtime (%)")
    save(fig, path)


def self_heating(th, path):
    """Runtime and end temperature at rated load against the cooling coefficient h, 0 and 25 C, reference and
    simulation. The isothermal model is the limit h -> infinity and is drawn at the right end of the axis."""
    rows = th["rows"]
    x_inf = 700.0
    ticks = [2, 5, 10, 25, 100, x_inf]

    def series(T_C, col):
        rr = sorted([r for r in rows if r["T_C"] == T_C and r["h"] is not None], key=lambda r: r["h"])
        iso = [r for r in rows if r["T_C"] == T_C and r["h"] is None][0]
        return [r["h"] for r in rr], [r[col] for r in rr], iso[col]

    def axis(ax):
        ax.set_xscale("log")
        ax.set_xlim(1.6, 2800)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["2", "5", "10", "25", "100", "∞"])
        ax.set_xlabel(r"$h$ (W m$^{-2}$ K$^{-1}$)")

    fig, axes = newfig(0.46, 2)
    ax = axes[0]
    for T_C, mk in ((0.0, "o"), (25.0, "s")):
        for col, color, ls, mfc in (("dfn_min", COLORS["reference"], "-", None),
                                    ("own_min", COLORS["simulation"], "--", "none")):
            h, t, t_iso = series(T_C, col)
            ax.plot(h, t, ls, marker=mk, color=color, ms=3.5, mfc=mfc)
            ax.plot([h[-1], x_inf], [t[-1], t_iso], ":", color=color, lw=1.0)
            ax.plot([x_inf], [t_iso], marker=mk, color=color, ms=3.5, mfc=mfc)
        ax.annotate(f"{T_C:.0f} °C", (x_inf, t_iso), xytext=(6, 0), textcoords="offset points", va="center")
    ax.axvline(10.0, color=LIGHT, lw=0.8, zorder=0)
    ax.annotate("reference $h$", (10.0, 0.30), xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", rotation=90, fontsize=FS - 1, color=GRAY, va="bottom")
    axis(ax)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.18 * (hi - lo), hi)
    ax.set_ylabel("Runtime at rated load (min)")
    ax.set_title("(a) Runtime")
    ax.plot([], [], "-", color=COLORS["reference"], label="Reference (DFN)")
    ax.plot([], [], "--", color=COLORS["simulation"], label="Simulation (SPMe)")
    ax.legend(loc="lower left")
    ax = axes[1]
    for T_C, mk in ((0.0, "o"), (25.0, "s")):
        h, T_end, T_iso = series(T_C, "dfn_T_peak_C")
        ax.plot(h, T_end, "-", marker=mk, color=COLORS["reference"], ms=3.5)
        ax.plot([h[-1], x_inf], [T_end[-1], T_iso], ":", color=COLORS["reference"], lw=1.0)
        ax.plot([x_inf], [T_iso], marker=mk, color=COLORS["reference"], ms=3.5)
        ax.annotate(f"{T_C:.0f} °C", (x_inf, T_iso), xytext=(6, 0), textcoords="offset points", va="center")
    axis(ax)
    ax.set_ylabel("Cell temperature at the end (°C)")
    ax.set_title("(b) End temperature (reference)")
    save(fig, path)


def model_forms(results_dir, path):
    """Error of three forms of the hand-written model against the reference, along the temperature sweep."""
    pts = {ds.key(d): d for d in rs.reference_points(os.path.join(results_dir, "reference.json"))}
    own = {ds.key(d): d for d in rs.load_json(os.path.join(results_dir, "model_points.json"))}
    fig, axes = newfig(0.42, 2, sharey=True)
    forms = [("SPM_isothermal", "Single particle, isothermal", COLORS["simulation"], ":"),
             ("SPMe_isothermal", "+ electrolyte, isothermal", COLORS["simulation"], "--"),
             ("SPMe_lumped", "+ electrolyte + self-heating", COLORS["simulation"], "-")]
    for ax, tag, title in zip(axes, ("sweep_T_0.6", "sweep_T_1.0"),
                              ("(a) 60 % of rated power, new cell", "(b) Rated power, new cell"), strict=True):
        ks = sorted([k for k in pts if k[0] == tag], key=lambda k: k[1])
        T = [k[1] for k in ks]
        for col, label, color, ls in forms:
            e = [100 * (own[k][col] / pts[k]["runtime_min"] - 1) for k in ks]
            ax.plot(T, e, ls, color=color, marker="o", ms=2.5, label=label)
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xlabel("Room temperature (°C)")
        ax.set_title(title)
    axes[0].set_ylabel("Runtime error (%)")
    axes[1].legend(loc="lower right")
    save(fig, path)


def sweeps(ml, path):
    """Reference and predictions along the temperature and state-of-health sweeps (seed 0)."""
    s0 = ml["seed0"]
    X, y, tag = np.array(s0["X"]), np.array(s0["y"]), np.array(s0["tag"])
    fig, axes = newfig(0.56, 2)
    panels = ((axes[0], "sweep_T_1.0", 0, "Room temperature (°C)", ds.NORMAL["T"], "(a) Rated load, new cell"),
              (axes[1], "sweep_soh", 2, "State of health", (ds.NORMAL["soh"][0], 1.0), "(b) 60 % load, 25 °C"))
    for ax, t, col, xlabel, band, title in panels:
        idx = np.where(tag == t)[0]
        idx = idx[np.argsort(X[idx, col])]
        x = X[idx, col]
        ax.axvspan(*band, color=COLORS["band"], alpha=0.10, lw=0)
        for key, label, color, ls in TOOLS:
            ax.plot(x, np.array(s0["preds"][key])[idx], ls, color=color, lw=1.3, label=label)
        ax.plot(x, y[idx], "o", color=COLORS["reference"], ms=4, zorder=5, label="Reference")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("Runtime (min)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, handlelength=2.0,
               columnspacing=1.2)
    save(fig, path)


def parity(ml, path):
    """Predicted against reference runtime, inside and outside the range (seed 0), square axes."""
    s0 = ml["seed0"]
    y, tag = np.array(s0["y"]), np.array(s0["tag"])
    test = np.array(s0["test_idx"])
    wide = np.where(tag == "wide")[0]
    show = [("Simulation (own model)", "(a) Simulation"), ("Gaussian process", "(b) Gaussian process"),
            ("Physics features + linear fit", "(c) Physics features")]
    fig, axes = newfig(0.38, 3)
    hi = 1.1 * y.max()
    colors = {t[0]: t[2] for t in TOOLS}
    for ax, (key, title) in zip(axes, show, strict=True):
        p = np.array(s0["preds"][key])
        ax.plot([1, hi], [1, hi], color=GRAY, lw=0.8)
        ax.scatter(y[wide], p[wide], s=10, facecolors="none", edgecolors=colors[key], lw=0.7, label="Outside")
        ax.scatter(y[test], p[test], s=10, color=colors[key], label="Inside")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(20, hi)
        ax.set_ylim(20, hi)
        ax.set_aspect("equal")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_locator(FixedLocator([20, 50, 100, 200]))
            a.set_major_formatter("{x:g}")
            a.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_xlabel("Reference (min)")
    axes[0].set_ylabel("Predicted runtime (min)")
    axes[0].legend(loc="upper left", handletextpad=0.2, borderpad=0.3)
    save(fig, path)


def coverage(ml, path):
    """Coverage of the 90 % conformal intervals inside and outside the training range."""
    acc = ml["accuracy"]
    keys = [k for k, *_ in TOOLS if "cover_in" in acc[k]] + ["Linear regression", "Polynomial (degree 3)",
                                                             "Random forest"]
    keys = list(dict.fromkeys(keys))
    names = {t[0]: t[1] for t in TOOLS}
    names.update({"Linear regression": "Linear regression", "Polynomial (degree 3)": "Polynomial",
                  "Random forest": "Random forest"})
    colors = {t[0]: t[2] for t in TOOLS}
    keys.sort(key=lambda k: acc[k]["cover_wide"][0])
    fig, ax = newfig(0.50)
    yy = np.arange(len(keys))
    inside = [100 * acc[k]["cover_in"][0] for k in keys]
    outside = [100 * acc[k]["cover_wide"][0] for k in keys]
    ax.barh(yy + 0.19, inside, 0.36, color=LIGHT)
    ax.barh(yy - 0.19, outside, 0.36, color=[colors.get(k, COLORS["ml"]) for k in keys])
    top = len(keys) - 1        # name the two bars of the top row instead of a legend
    ax.annotate("inside the training range", (0, top + 0.19), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=FS - 1)
    ax.annotate("outside", (0, top - 0.19), xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=FS - 1, color="white")
    ax.axvline(90, color=INK, lw=0.9, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([names[k] for k in keys])
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.set_xlabel("Test points inside the 90 % interval (%)")
    ax.set_xlim(0, 100)
    save(fig, path)


def runtime_map(grid, path):
    """Simulated runtime over temperature and load (new cell), training range outlined."""
    T, L = np.array(grid["T_C"]), np.array(grid["load"])
    G = np.array(grid["runtime_min"])[:, :, 0]
    fig, ax = newfig(0.62)
    levels = [10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300]
    cs = ax.contourf(T, 100 * L, G.T, levels=levels, cmap="viridis", extend="both")
    cl = ax.contour(T, 100 * L, G.T, levels=levels, colors="white", linewidths=0.6)
    ax.clabel(cl, fmt="%g", fontsize=FS - 1.5)
    lo, hi = ds.NORMAL["T"], ds.NORMAL["load"]
    ax.add_patch(plt.Rectangle((lo[0], 100 * hi[0]), lo[1] - lo[0], 100 * (hi[1] - hi[0]), fill=False, ec="white",
                               lw=1.4, ls="--"))
    ax.annotate("training range", (lo[1], 100 * hi[1]), xytext=(3, 3), textcoords="offset points", color="white",
                fontsize=FS - 0.5)
    fig.colorbar(cs, ax=ax, label="Runtime (min)")
    ax.set_xlabel("Room temperature (°C)")
    ax.set_ylabel("Load (% of rated power)")
    save(fig, path)


def convergence(conv, path):
    """Radial convergence of the single-particle model, ours against PyBaMM's (0 C, 2C)."""
    sp = conv["spm"]
    fig, ax = newfig(0.58)
    ref_t = sp["pybamm"][-1]["runtime_s"]
    for rows, label, color, mk in ((sp["own"], "This code (Kirchhoff faces)", COLORS["simulation"], "o"),
                                   (sp["pybamm"], "PyBaMM SPM", COLORS["reference"], "s")):
        n = [r["n_r"] for r in rows]
        e = [max(abs(r["runtime_s"] / ref_t - 1) * 100, 1e-3) for r in rows]
        ax.plot(n, e, "-", marker=mk, ms=4, color=color, label=label)
    ax.axvline(20, color=LIGHT, lw=0.8, zorder=0)
    ax.annotate("PyBaMM default", (20, 0.05), xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", rotation=90, fontsize=FS - 1, color=GRAY, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Radial points per particle")
    ax.set_ylabel("Runtime difference from the\nfinest PyBaMM run (%)")
    ax.legend()
    save(fig, path)


def runaway(results_dir, path):
    """Oven tests of the three chemistries and propagation along a row."""
    from . import thermal_runaway as tr
    run = rs.load_json(os.path.join(results_dir, "verify_runaway.json"))
    sc = rs.load_json(os.path.join(results_dir, "runaway_scenarios.json"))
    fig, axes = newfig(0.46, 2)
    ax = axes[0]
    shades = {"LCO": "0.1", "NMC111": "0.42", "LFP": "0.65"}
    for ch in (tr.LCO, tr.NMC, tr.LFP):
        crit = run["oven"][f"{ch.name}_critical_C"]
        for dT, ls in ((+2.0, "-"), (-2.0, ":")):
            t, y, _ = tr.oven_test(ch, crit + dT + 273.15, t_end=10800.0)
            T = y[:, 0] - 273.15
            ax.plot(t / 60, T, ls, color=shades[ch.name], lw=1.2)
            if dT > 0:      # name the chemistry above its runaway peak
                k = int(np.argmax(T))
                right = ch.name == "LCO"      # its neighbours leave no room above the peak
                ax.annotate(f"{ch.name}\nonset {crit:.0f} °C", (t[k] / 60, T[k]), xytext=(5, -6) if right else (0, 4),
                            textcoords="offset points", fontsize=FS - 1, color=shades[ch.name],
                            va="center" if right else "bottom", ha="left" if right else "center", linespacing=1.0)
    ax.set_xlabel("Time in the oven (min)")
    ax.set_ylabel("Cell temperature (°C)")
    ax.set_ylim(0, 650)
    ax.set_title("(a) Oven tests, onset ± 2 K")
    ax = axes[1]
    for name, mk, ms, mfc in (("NMC111", "s", 3.5, None), ("LFP", "^", 3.5, None), ("LCO", "o", 6.0, "none")):
        rr = [r for r in sc["propagation"] if r["chem"] == name]
        ax.plot([r["G"] for r in rr], [r["n_runaway"] for r in rr], "-", marker=mk, ms=ms, mfc=mfc,
                color=shades[name], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Conductance to a neighbour (W K$^{-1}$)")
    ax.set_ylabel("Cells in runaway (of 6)")
    ax.set_title("(b) Row of six, first at 250 °C")
    ax.legend(loc="lower right")
    save(fig, path)
