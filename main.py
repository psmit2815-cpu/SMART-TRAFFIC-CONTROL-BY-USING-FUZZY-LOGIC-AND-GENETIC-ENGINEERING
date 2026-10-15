"""
main.py - diagnostic runner & benchmark for the Smart Traffic Signal project
============================================================================
    python main.py                 full run (GA: 24 individuals x 30 generations)
    python main.py --quick         small GA (fast demo)
    python main.py --skip-ga       re-use results/ga_best_params.json
It executes: 10 diagnostic test cases, Mamdani-vs-Sugeno dataset benchmark,
GA optimisation, a 10-scenario traffic simulation benchmark, and regenerates
every plot / CSV under results/.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from data.generate_dataset import generate, OUT as DATA_CSV
from src.fuzzy_system import MamdaniFIS, DEFAULT_PARAMS, GENE_NAMES
from src.sugeno_model import SugenoFIS
from src.controllers import FixedTimeController, QueueProportionalController, FISController
from src.traffic_simulator import SCENARIOS, run_simulation
from src.genetic_optimizer import optimise_membership_functions, TRAIN_SCENARIOS
from src import utils as U

R = U.RESULTS
TEST_SEEDS = (1, 2, 3, 4, 5)

TEST_CASES = [   # name, q_this, q_other, arrival (veh/min)
    ("Empty junction (night)", 0, 0, 0),
    ("Light traffic both sides", 5, 5, 6),
    ("Moderate balanced", 18, 18, 16),
    ("Heavy on green side, light opposing", 45, 8, 30),
    ("Light on green side, heavy opposing", 6, 48, 8),
    ("Heavy both sides (saturation)", 52, 52, 34),
    ("Medium queue, high arrivals", 25, 15, 32),
    ("Large queue, low arrivals", 42, 20, 6),
    ("Small queue, high arrivals", 8, 10, 30),
    ("Everything medium", 25, 25, 20),
]


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def diagnostics(mam, sug):
    banner("1. DIAGNOSTIC TEST CASES  (Mamdani vs Sugeno green time)")
    rows = []
    for name, a, b, c in TEST_CASES:
        gm, gs = mam.infer(a, b, c), sug.infer(a, b, c)
        cat, _ = U.classify_green(gm)
        rows.append({"case": name, "Q_THIS": a, "Q_OTHER": b, "ARRIVAL": c,
                     "mamdani_s": round(gm, 1), "sugeno_s": round(gs, 1), "category": cat})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(R, "diagnostic_cases.csv"), index=False)
    res = mam.infer(30, 12, 20, details=True)
    U.plot_defuzzification(res, os.path.join(R, "comparison_plots", "defuzzification_example.png"),
                           "Centroid defuzzification: Q_THIS=30, Q_OTHER=12, ARRIVAL=20")
    return df


def dataset_benchmark(mam, sug):
    banner("2. DATASET BENCHMARK  (3,000 synthetic states vs crisp reference heuristic)")
    df = pd.read_csv(DATA_CSV) if os.path.exists(DATA_CSV) else generate()
    X = df[["q_this", "q_other", "arrival_this"]].values
    ref = df["reference_green"].values
    out = {}
    for name, fis in (("Mamdani", mam), ("Sugeno", sug)):
        t0 = time.perf_counter()
        pred = np.array([fis.infer(*x) for x in X])
        us = (time.perf_counter() - t0) / len(X) * 1e6
        out[name] = {"MAE_s": np.mean(np.abs(pred - ref)), "RMSE_s": np.sqrt(np.mean((pred - ref) ** 2)),
                     "corr_with_reference": np.corrcoef(pred, ref)[0, 1], "us_per_inference": us}
        if name == "Mamdani":
            mp = pred
        else:
            sp = pred
    out["Mamdani_vs_Sugeno"] = {"MAE_s": np.mean(np.abs(mp - sp)),
                                "RMSE_s": np.sqrt(np.mean((mp - sp) ** 2)),
                                "corr_with_reference": np.corrcoef(mp, sp)[0, 1], "us_per_inference": np.nan}
    tab = pd.DataFrame(out).T.round(3)
    print(tab.to_string())
    tab.to_csv(os.path.join(R, "dataset_benchmark.csv"))
    return tab


def run_ga(args):
    banner("3. GENETIC ALGORITHM OPTIMISATION  (tuning 14 membership-function genes)")
    hist_file = os.path.join(R, "ga_history.json")
    if args.skip_ga and os.path.exists(U.GA_PARAMS_FILE) and os.path.exists(hist_file):
        print("  --skip-ga: loading saved GA result")
        with open(hist_file) as f:
            info = json.load(f)
        return U.load_ga_params(), info
    pop, gens = (12, 10) if args.quick else (args.pop, args.gens)
    print(f"  population={pop}, generations={gens}, training scenarios={TRAIN_SCENARIOS}")
    res = optimise_membership_functions(pop_size=pop, generations=gens, verbose=True)
    info = {"baseline_fitness": res["baseline_fitness"], "best_fitness": res["best_fitness"],
            "evaluations": res["evaluations"], "runtime_s": res["runtime"], "history": res["history"],
            "pop_size": pop, "generations": gens, "train_scenarios": TRAIN_SCENARIOS,
            "default_params": [float(x) for x in DEFAULT_PARAMS]}
    U.save_ga_params(res["best_params"], {"gene_names": GENE_NAMES, "best_fitness": res["best_fitness"]})
    with open(hist_file, "w") as f:
        json.dump(info, f, indent=2)
    return res["best_params"], info


def ga_report(best, info, mam_def, mam_ga):
    gain = 100 * (info["baseline_fitness"] - info["best_fitness"]) / info["baseline_fitness"]
    print(f"\n  training fitness: default {info['baseline_fitness']:.2f} s/veh -> GA {info['best_fitness']:.2f} s/veh "
          f"({gain:.1f}% lower)   | {info['evaluations']} evaluations, {info['runtime_s']:.0f} s")
    tab = pd.DataFrame({"gene": GENE_NAMES, "default": DEFAULT_PARAMS.round(1), "GA_tuned": mam_ga.params.round(1)})
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(R, "ga_parameters.csv"), index=False)
    U.plot_ga_convergence(info["history"], os.path.join(R, "comparison_plots", "ga_convergence.png"))
    U.plot_membership(list(mam_ga.inputs.values()) + [mam_ga.output], "GA-tuned membership functions",
                      os.path.join(R, "membership_plots", "membership_ga_tuned.png"))


def simulation_benchmark(controllers):
    banner("4. TRAFFIC SIMULATION BENCHMARK  (10 scenarios x 5 controllers x 5 test seeds, 1 h each)")
    rows = []
    for sn in SCENARIOS:
        for c in controllers:
            for sd in TEST_SEEDS:
                r = run_simulation(sn, c, seed=sd)
                rows.append({"scenario": sn, "controller": c.name, "seed": sd, "avg_delay": r.avg_delay,
                             "avg_queue": r.avg_queue, "max_queue": r.max_queue, "throughput": r.throughput,
                             "final_queue": r.final_queue, "avg_green": r.avg_green})
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(R, "simulation_runs.csv"), index=False)
    summ = raw.groupby(["scenario", "controller"]).mean(numeric_only=True).drop(columns="seed").round(2).reset_index()
    summ["in_GA_training"] = summ["scenario"].isin(TRAIN_SCENARIOS)
    summ.to_csv(os.path.join(R, "simulation_summary.csv"), index=False)
    delay = summ.pivot(index="scenario", columns="controller", values="avg_delay")
    delay = delay[[c.name for c in controllers]]
    print("\nAverage delay per vehicle (s), mean of 5 test seeds:")
    print(delay.round(1).to_string())
    overall = raw.groupby("controller")[["avg_delay", "avg_queue", "throughput", "max_queue"]].mean().round(2)
    overall = overall.loc[[c.name for c in controllers]]
    base = overall.loc[controllers[0].name, "avg_delay"]
    overall["delay_vs_fixed_%"] = ((overall["avg_delay"] - base) / base * 100).round(1)
    overall["scenarios_won"] = delay.idxmin(axis=1).value_counts().reindex(overall.index).fillna(0).astype(int)
    print("\nOverall (mean over all scenarios and seeds):")
    print(overall.to_string())
    overall.to_csv(os.path.join(R, "overall_summary.csv"))
    U.plot_delay_comparison(raw, os.path.join(R, "comparison_plots", "delay_comparison.png"))
    return raw, overall


def timeseries_plot(controllers):
    res = [run_simulation("S06_morning_peak_ns", c, seed=3, record=True) for c in controllers[:1] + controllers[2:3] + controllers[4:5]]
    U.plot_timeseries(res, os.path.join(R, "comparison_plots", "queue_timeseries_S06.png"),
                      "Morning-peak scenario (S06): queue length over time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small GA for a fast demo")
    ap.add_argument("--skip-ga", action="store_true", help="re-use saved GA result")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    if not os.path.exists(DATA_CSV):
        generate()

    mam, sug = MamdaniFIS(), SugenoFIS()
    print(f"Mamdani FIS ready: {len(mam.rules)} rules, 3 inputs, 1 output (green time 10-60 s)")
    U.plot_membership(list(mam.inputs.values()) + [mam.output], "Default (expert) membership functions",
                      os.path.join(R, "membership_plots", "membership_default.png"))
    diagnostics(mam, sug)
    dataset_benchmark(mam, sug)

    best, info = run_ga(args)
    mam_ga = MamdaniFIS(best)
    ga_report(best, info, mam, mam_ga)

    U.plot_surface(mam, os.path.join(R, "surface_plots", "surface_default_qthis_qother.png"), "Q_THIS", "Q_OTHER", fixed_arrival=16)
    U.plot_surface(mam, os.path.join(R, "surface_plots", "surface_default_qthis_arrival.png"), "Q_THIS", "ARRIVAL", fixed_q_other=20)
    U.plot_surface(mam_ga, os.path.join(R, "surface_plots", "surface_ga_qthis_qother.png"), "Q_THIS", "Q_OTHER", fixed_arrival=16)

    controllers = [FixedTimeController(30), QueueProportionalController(),
                   FISController(mam, "Fuzzy Mamdani (expert)"), FISController(sug, "Fuzzy Sugeno (expert)"),
                   FISController(mam_ga, "Fuzzy-GA Mamdani (tuned)")]
    simulation_benchmark(controllers)
    timeseries_plot(controllers)
    banner("DONE - plots, CSVs and GA parameters saved in results/")


if __name__ == "__main__":
    main()
