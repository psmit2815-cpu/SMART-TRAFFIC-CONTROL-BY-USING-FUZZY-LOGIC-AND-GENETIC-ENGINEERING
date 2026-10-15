"""
utils.py
========
Green-time categorisation, parameter I/O and all plotting helpers.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
GA_PARAMS_FILE = os.path.join(RESULTS, "ga_best_params.json")

# Green-time bands (like the "risk categories" of a classic FIS report)
GREEN_CATEGORIES = [
    (10, 20, "Very Short", "Light demand - release the junction quickly."),
    (20, 30, "Short", "Low-moderate demand - short phase keeps cycle compact."),
    (30, 40, "Medium", "Balanced demand - standard phase length."),
    (40, 50, "Long", "Heavy demand - extend green to clear the queue."),
    (50, 60, "Very Long", "Saturated approach - maximum green allowed."),
]


def classify_green(g):
    for lo, hi, name, action in GREEN_CATEGORIES:
        if g < hi:
            return name, action
    return GREEN_CATEGORIES[-1][2], GREEN_CATEGORIES[-1][3]


def congestion_level(q_total):
    """Simple label for the total queue on the junction (vehicles)."""
    if q_total < 10:
        return "Free flow"
    if q_total < 30:
        return "Moderate"
    if q_total < 60:
        return "Congested"
    return "Heavily congested"


def save_ga_params(params, extra=None, path=GA_PARAMS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"params": [float(x) for x in params]}
    data.update(extra or {})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_ga_params(path=GA_PARAMS_FILE):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return np.array(json.load(f)["params"], dtype=float)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ plotting
def plot_membership(variables, title, path):
    """variables: list of LinguisticVariable -> one row of subplots."""
    fig, axes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 3.4))
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, variables):
        x = v.grid(500)
        for t, mf in v.terms.items():
            ax.plot(x, mf(x), lw=2, label=t)
        ax.set_title(f"{v.name} ({v.unit})" if v.unit else v.name)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(v.unit or "")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("membership degree")
    fig.suptitle(title, fontweight="bold")
    return _save(fig, path)


def plot_defuzzification(res, path, title="Mamdani aggregation and centroid defuzzification"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for t, mu in res["output_sets"].items():
        ax.plot(res["y"], mu, "--", lw=0.8, alpha=0.5, label=t)
    ax.fill_between(res["y"], res["mu"], color="tab:orange", alpha=0.45, label="aggregated output")
    ax.axvline(res["green"], color="crimson", lw=2, label=f"centroid = {res['green']:.1f} s")
    ax.set_xlabel("green time (s)")
    ax.set_ylabel("membership")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_surface(fis, path, x="Q_THIS", y="Q_OTHER", fixed_arrival=16.0, fixed_q_other=20.0, fixed_q_this=20.0):
    """Control surface (heat-map + contours) of green time over two inputs."""
    ranges = {"Q_THIS": (0, 60), "Q_OTHER": (0, 60), "ARRIVAL": (0, 40)}
    xs = np.linspace(*ranges[x], 41)
    ys = np.linspace(*ranges[y], 41)
    Z = np.zeros((len(ys), len(xs)))
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            v = {"Q_THIS": fixed_q_this, "Q_OTHER": fixed_q_other, "ARRIVAL": fixed_arrival}
            v[x], v[y] = xv, yv
            Z[i, j] = fis.infer(v["Q_THIS"], v["Q_OTHER"], v["ARRIVAL"])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.contourf(xs, ys, Z, levels=20, cmap="RdYlGn")
    cs = ax.contour(xs, ys, Z, levels=8, colors="k", linewidths=0.4)
    ax.clabel(cs, fontsize=6, fmt="%.0f")
    fixed = {"Q_THIS": fixed_q_this, "Q_OTHER": fixed_q_other, "ARRIVAL": fixed_arrival}
    note = ", ".join(f"{k}={v:g}" for k, v in fixed.items() if k not in (x, y))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Green-time control surface ({note})", fontsize=9)
    fig.colorbar(im, label="green time (s)")
    return _save(fig, path)


def plot_ga_convergence(history, path):
    g = np.arange(len(history["best"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(g, history["best"], lw=2.2, label="best")
    ax.plot(g, history["mean"], lw=1.4, label="population mean")
    ax.plot(g, history["worst"], lw=0.9, alpha=0.6, label="worst")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness = mean delay (s/veh)  [lower is better]")
    ax.set_title("GA convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_delay_comparison(df, path):
    """df: columns scenario, controller, avg_delay (mean over seeds)."""
    piv = df.pivot_table(index="scenario", columns="controller", values="avg_delay", aggfunc="mean")
    order = ["Fixed-Time (30s)", "Queue-Proportional (crisp)", "Fuzzy Mamdani (expert)",
             "Fuzzy Sugeno (expert)", "Fuzzy-GA Mamdani (tuned)"]
    piv = piv[[c for c in order if c in piv.columns] + [c for c in piv.columns if c not in order]]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    piv.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel("average delay (s / vehicle)")
    ax.set_xlabel("")
    ax.set_title("Average vehicle delay per scenario (mean of test seeds, lower is better)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_timeseries(results, path, title="Total queue length over time"):
    """results: list of SimResult with history recorded."""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.2 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        q = r.history["q"]
        t = np.arange(len(q))
        ax.plot(t, q[:, 0] + q[:, 1], label="N+S queue", lw=1)
        ax.plot(t, q[:, 2] + q[:, 3], label="E+W queue", lw=1)
        ax.set_ylabel("veh")
        ax.set_title(f"{r.controller}  (avg delay {r.avg_delay:.1f} s)", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def draw_intersection(ax, q, light, t):
    """Draw a schematic intersection with queue bars and signal lights."""
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-1.6, -10), 3.2, 20, color="#444"))
    ax.add_patch(plt.Rectangle((-10, -1.6), 20, 3.2, color="#444"))
    ax.add_patch(plt.Rectangle((-1.6, -1.6), 3.2, 3.2, color="#666"))
    ns_green, ew_green = light == 1, light == 2
    col = lambda on: "#2ecc71" if on else ("#f1c40f" if light == 3 else "#e74c3c")
    ax.add_patch(plt.Circle((-2.3, 2.3), 0.6, color=col(ns_green)))   # N head
    ax.add_patch(plt.Circle((2.3, -2.3), 0.6, color=col(ns_green)))   # S head
    ax.add_patch(plt.Circle((-2.3, -2.3), 0.6, color=col(ew_green)))  # W head
    ax.add_patch(plt.Circle((2.3, 2.3), 0.6, color=col(ew_green)))    # E head
    cap = 40.0
    n, s, e, w = (int(v) for v in q)
    ax.add_patch(plt.Rectangle((-1.0, 2.0), 0.8, min(n / cap, 1) * 7.5, color="tab:blue"))
    ax.add_patch(plt.Rectangle((0.2, -2.0 - min(s / cap, 1) * 7.5), 0.8, min(s / cap, 1) * 7.5, color="tab:blue"))
    ax.add_patch(plt.Rectangle((2.0, 0.2), min(e / cap, 1) * 7.5, 0.8, color="tab:orange"))
    ax.add_patch(plt.Rectangle((-2.0 - min(w / cap, 1) * 7.5, -1.0), min(w / cap, 1) * 7.5, 0.8, color="tab:orange"))
    ax.text(0, 9.3, f"N: {n}", ha="center", fontsize=9)
    ax.text(0, -9.7, f"S: {s}", ha="center", fontsize=9)
    ax.text(9.2, 0, f"E: {e}", ha="center", fontsize=9)
    ax.text(-9.2, 0, f"W: {w}", ha="center", fontsize=9)
    state = {0: "ALL RED", 1: "N-S GREEN", 2: "E-W GREEN", 3: "YELLOW / ALL-RED"}[int(light)]
    ax.set_title(f"t = {t} s  |  {state}", fontsize=10)
