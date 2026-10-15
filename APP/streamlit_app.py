"""
streamlit_app.py - Smart Traffic Signal Control dashboard (software only)
Run:  streamlit run app/streamlit_app.py
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.fuzzy_system import MamdaniFIS, DEFAULT_PARAMS, GENE_NAMES
from src.sugeno_model import SugenoFIS
from src.fuzzy_rules import RULES, rule_text, INPUT_NAMES
from src.controllers import FixedTimeController, QueueProportionalController, FISController
from src.traffic_simulator import SCENARIOS, run_simulation
from src.genetic_optimizer import optimise_membership_functions
from src import utils as U

st.set_page_config(page_title="Smart Traffic Signal - Fuzzy + GA", page_icon="🚦", layout="wide")


@st.cache_resource
def get_models(ga_key=0):
    ga = U.load_ga_params()
    return MamdaniFIS(), SugenoFIS(), (MamdaniFIS(ga) if ga is not None else None)


if "ga_version" not in st.session_state:
    st.session_state.ga_version = 0
mam, sug, mam_ga = get_models(st.session_state.ga_version)


def controller_options():
    opts = {"Fixed-Time (30 s)": FixedTimeController(30),
            "Queue-Proportional (crisp)": QueueProportionalController(),
            "Fuzzy Mamdani (expert)": FISController(mam, "Fuzzy Mamdani (expert)"),
            "Fuzzy Sugeno (expert)": FISController(sug, "Fuzzy Sugeno (expert)")}
    if mam_ga is not None:
        opts["Fuzzy-GA Mamdani (tuned)"] = FISController(mam_ga, "Fuzzy-GA Mamdani (tuned)")
    return opts


st.title("🚦 Smart Traffic Signal Control using Fuzzy Logic & Genetic Algorithms")
st.caption("Software-only project: virtual detectors + simulated 4-way intersection replace all hardware.")

tabs = st.tabs(["🎛️ Signal Controller", "🚦 Intersection Simulation", "📊 Controller Comparison",
                "📈 Control Surface", "🧬 GA Optimisation", "ℹ️ About"])

# ----------------------------------------------------------------- TAB 1
with tabs[0]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Detector inputs")
        q_this = st.slider("Queue on approach about to get GREEN (veh)", 0, 60, 30)
        q_other = st.slider("Queue on opposing approach (veh)", 0, 60, 12)
        arr = st.slider("Arrival rate on green approach (veh/min)", 0, 40, 20)
        res = mam.infer(q_this, q_other, arr, details=True)
        sres = sug.infer(q_this, q_other, arr)
        cat, action = U.classify_green(res["green"])
        m1, m2 = st.columns(2)
        m1.metric("Mamdani green time", f"{res['green']:.1f} s")
        m2.metric("Sugeno green time", f"{sres:.1f} s")
        if mam_ga is not None:
            st.metric("GA-tuned Mamdani green time", f"{mam_ga.infer(q_this, q_other, arr):.1f} s")
        st.info(f"**{cat}** - {action}")
        st.progress(float((res["green"] - 10) / 50))
        st.caption(f"Junction state: **{U.congestion_level(q_this + q_other)}**")
    with c2:
        st.subheader("Mamdani aggregation & centroid")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for t, mu in res["output_sets"].items():
            ax.plot(res["y"], mu, "--", lw=0.8, alpha=0.5, label=t)
        ax.fill_between(res["y"], res["mu"], color="tab:orange", alpha=0.45)
        ax.axvline(res["green"], color="crimson", lw=2)
        ax.set_xlabel("green time (s)"); ax.set_ylabel("membership"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        st.pyplot(fig); plt.close(fig)
    st.subheader("Fuzzified inputs and activated rules")
    f1, f2 = st.columns(2)
    with f1:
        st.dataframe(pd.DataFrame(res["fuzzified"]).T.round(3), width="stretch")
    with f2:
        act = [{"Rule": r.id, "Strength": round(w, 3), "Rule text": rule_text(r)}
               for r, w in zip(RULES, res["strengths"]) if w > 0]
        st.dataframe(pd.DataFrame(act).sort_values("Strength", ascending=False),
                     width="stretch", hide_index=True)
    st.subheader("Membership functions")
    which = st.radio("Parameter set", ["Expert (default)"] + (["GA-tuned"] if mam_ga is not None else []), horizontal=True)
    fis = mam if which.startswith("Expert") else mam_ga
    vars_ = list(fis.inputs.values()) + [fis.output]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, v in zip(axes, vars_):
        x = v.grid(300)
        for t, mf in v.terms.items():
            ax.plot(x, mf(x), lw=2, label=t)
        ax.set_title(v.name); ax.grid(alpha=.3); ax.legend(fontsize=6)
    st.pyplot(fig); plt.close(fig)

# ----------------------------------------------------------------- TAB 2
with tabs[1]:
    st.subheader("Simulate one hour at a 4-way intersection")
    a, b, c, d = st.columns(4)
    scen = a.selectbox("Traffic scenario", list(SCENARIOS), index=5)
    opts = controller_options()
    ctrl_name = b.selectbox("Controller", list(opts), index=2)
    seed = c.number_input("Random seed", 0, 9999, 3)
    dur = d.select_slider("Duration (s)", [600, 1200, 1800, 3600], value=1800)
    st.caption(SCENARIOS[scen].description)
    if st.button("▶ Run simulation", type="primary"):
        st.session_state["sim"] = run_simulation(scen, opts[ctrl_name], seed=int(seed), duration=int(dur), record=True)
    r = st.session_state.get("sim")
    if r is not None:
        k = st.columns(5)
        k[0].metric("Avg delay", f"{r.avg_delay:.1f} s/veh")
        k[1].metric("Avg queue", f"{r.avg_queue:.1f} veh")
        k[2].metric("Max queue", r.max_queue)
        k[3].metric("Throughput", f"{r.throughput:.0f} veh/h")
        k[4].metric("Avg green", f"{r.avg_green:.1f} s")
        t = st.slider("Replay time (s)", 0, len(r.history["q"]) - 1, 0, step=5)
        left, right = st.columns([1, 2])
        with left:
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            U.draw_intersection(ax, r.history["q"][t], r.history["light"][t], t)
            st.pyplot(fig); plt.close(fig)
        with right:
            q = r.history["q"]
            fig, ax = plt.subplots(figsize=(8, 3.6))
            ax.plot(q[:, 0] + q[:, 1], label="N+S queue"); ax.plot(q[:, 2] + q[:, 3], label="E+W queue")
            ax.axvline(t, color="k", lw=1)
            ax.set_xlabel("time (s)"); ax.set_ylabel("vehicles"); ax.legend(); ax.grid(alpha=.3)
            st.pyplot(fig); plt.close(fig)
        with st.expander("Signal timing log (first 20 phases)"):
            st.dataframe(pd.DataFrame(r.phase_log[:20], columns=["start (s)", "phase", "green (s)"]), hide_index=True)

# ----------------------------------------------------------------- TAB 3
with tabs[2]:
    st.subheader("Compare every controller on the same traffic")
    scen2 = st.selectbox("Scenario", list(SCENARIOS), key="cmp_scen")
    n_seeds = st.slider("Number of random seeds", 1, 5, 3)
    if st.button("Run comparison"):
        rows = []
        for name, ctrl in controller_options().items():
            rs = [run_simulation(scen2, ctrl, seed=s, duration=3600) for s in range(1, n_seeds + 1)]
            rows.append({"Controller": name, "Avg delay (s/veh)": np.mean([x.avg_delay for x in rs]),
                         "Avg queue": np.mean([x.avg_queue for x in rs]),
                         "Max queue": np.mean([x.max_queue for x in rs]),
                         "Throughput (veh/h)": np.mean([x.throughput for x in rs])})
        df = pd.DataFrame(rows).round(2)
        st.dataframe(df, hide_index=True, width="stretch")
        st.bar_chart(df.set_index("Controller")["Avg delay (s/veh)"])
    summ = os.path.join(U.RESULTS, "overall_summary.csv")
    if os.path.exists(summ):
        st.markdown("**Saved benchmark from `python main.py` (10 scenarios x 5 test seeds):**")
        st.dataframe(pd.read_csv(summ), hide_index=True, width="stretch")
        img = os.path.join(U.RESULTS, "comparison_plots", "delay_comparison.png")
        if os.path.exists(img):
            st.image(img)

# ----------------------------------------------------------------- TAB 4
with tabs[3]:
    st.subheader("Green-time control surface")
    s1, s2, s3 = st.columns(3)
    xv = s1.selectbox("X axis", INPUT_NAMES, 0)
    yv = s2.selectbox("Y axis", INPUT_NAMES, 1)
    use = s3.radio("FIS", ["Expert"] + (["GA-tuned"] if mam_ga is not None else []), horizontal=True)
    fixed = {"Q_THIS": st.slider("Fixed Q_THIS", 0, 60, 20), "Q_OTHER": st.slider("Fixed Q_OTHER", 0, 60, 20),
             "ARRIVAL": st.slider("Fixed ARRIVAL", 0, 40, 16)}
    if xv == yv:
        st.warning("Choose two different inputs.")
    else:
        path = os.path.join(U.RESULTS, "surface_plots", "_live_surface.png")
        U.plot_surface(mam if use == "Expert" else mam_ga, path, xv, yv, fixed["ARRIVAL"], fixed["Q_OTHER"], fixed["Q_THIS"])
        st.image(path)

# ----------------------------------------------------------------- TAB 5
with tabs[4]:
    st.subheader("Genetic Algorithm tuning of membership functions")
    hist_file = os.path.join(U.RESULTS, "ga_history.json")
    if os.path.exists(hist_file):
        info = json.load(open(hist_file))
        g1, g2, g3 = st.columns(3)
        g1.metric("Training delay - expert", f"{info['baseline_fitness']:.2f} s")
        g2.metric("Training delay - GA", f"{info['best_fitness']:.2f} s")
        g3.metric("Fitness evaluations", info["evaluations"])
        st.line_chart(pd.DataFrame(info["history"]))
        if mam_ga is not None:
            st.dataframe(pd.DataFrame({"gene": GENE_NAMES, "expert": DEFAULT_PARAMS.round(1),
                                       "GA-tuned": mam_ga.params.round(1)}), hide_index=True)
    else:
        st.info("No saved GA result yet - run `python main.py` or use the button below.")
    st.markdown("**Run a fresh (small) GA from here**")
    p, gn = st.columns(2)
    pop = p.slider("Population", 8, 30, 12)
    gens = gn.slider("Generations", 5, 30, 10)
    if st.button("🧬 Run GA now"):
        with st.spinner("Evolving membership functions (runs the traffic simulator for every candidate)..."):
            out = optimise_membership_functions(pop_size=pop, generations=gens, verbose=False)
        U.save_ga_params(out["best_params"], {"gene_names": GENE_NAMES, "best_fitness": out["best_fitness"]})
        json.dump({"baseline_fitness": out["baseline_fitness"], "best_fitness": out["best_fitness"],
                   "evaluations": out["evaluations"], "runtime_s": out["runtime"], "history": out["history"],
                   "pop_size": pop, "generations": gens}, open(hist_file, "w"), indent=2)
        st.session_state.ga_version += 1
        get_models.clear()
        st.success(f"Done: training delay {out['baseline_fitness']:.2f} -> {out['best_fitness']:.2f} s/veh. Reloading...")
        st.rerun()

# ----------------------------------------------------------------- TAB 6
with tabs[5]:
    st.markdown("""
**Problem.** Fixed-time signals ignore real traffic, so queues build on busy approaches while green is wasted on empty ones.

**Solution.** A Mamdani fuzzy controller reads three virtual-detector values - queue on the approach about to get green,
queue on the opposing approach, and arrival rate - and outputs the green time (10-60 s) using 27 IF-THEN rules.
A real-coded Genetic Algorithm then tunes 14 membership-function parameters by minimising the average vehicle delay
measured in the traffic simulator.

**Software-only.** No sensors, microcontrollers or signal heads: arrivals are Poisson, departures follow saturation flow,
and detectors are simulated inside `src/traffic_simulator.py`.
""")
    st.table(pd.DataFrame([{"Green time (s)": f"{lo}-{hi}", "Category": n, "Meaning": a} for lo, hi, n, a in U.GREEN_CATEGORIES]))
