# Smart Traffic Signal Control Using Fuzzy Logic and Genetic Algorithms

**Semester 5 Electronics & Instrumentation Engineering Final Project - Software-Only Implementation**

Patel Smit Kalpesh | Enrollment 240170147018 | VGEC

An intelligent traffic-signal controller built using a **Mamdani Fuzzy Inference System (FIS)**, a **Zero-Order Sugeno FIS comparison**, and a **Real-Coded Genetic Algorithm (GA)** that tunes the membership functions. The whole project runs in software: a Python **traffic simulator with virtual detectors replaces every hardware component** (no sensors, microcontrollers or signal heads needed).

---

## 🌟 Key Features

1. **Core Mamdani FIS Engine**: Pure NumPy implementation of fuzzification, min-max implication/aggregation and Centroid (COA) defuzzification mapping 3 detector inputs to a **Green Time (10-60 s)**.
2. **3 Traffic Input Variables** (read from virtual detectors):
   - Q_THIS - queue on the approach about to get green `[0 to 60 veh]`
   - Q_OTHER - queue on the opposing approach `[0 to 60 veh]`
   - ARRIVAL - arrival rate on the green approach `[0 to 40 veh/min]`
3. **27 Mamdani Rules**: a full 3 x 3 x 3 rule grid, grouped into Light, Moderate and Heavy demand on the approach receiving green.
4. **Sugeno Model Benchmark**: side-by-side comparison with a Zero-Order Sugeno FIS that re-uses the same rules.
5. **Genetic Algorithm (GA) Hybrid Optimizer**: real-coded GA tuning **14 membership-function genes**; fitness = average vehicle delay measured in the simulator (Units 4-6 syllabus module).
6. **Software Traffic Simulator (replaces hardware)**: 4-way, 2-phase intersection, Poisson arrivals, saturation-flow departures, yellow/all-red lost time, 10 traffic scenarios (light, heavy, peak-hour, surge, one-sided...).
7. **Controller Benchmark**: Fixed-Time vs crisp Queue-Proportional vs Fuzzy Mamdani vs Fuzzy Sugeno vs Fuzzy-GA.
8. **Interactive Streamlit Dashboard**: live green-time calculator, activated-rule inspection, centroid curve, control-surface heat-maps, animated intersection replay, controller comparison and in-app GA run.

---

## 📁 Project Structure

```
traffic-signal-fuzzy/
│
├── data/
│   ├── generate_dataset.py        # Synthetic dataset generator script
│   └── traffic_data.csv           # 3,000 synthetic intersection states with reference green time
│
├── src/
│   ├── __init__.py
│   ├── membership_functions.py    # Triangular, Trapezoidal, Gaussian MFs math & classes
│   ├── fuzzification.py           # LinguisticVariable + fuzzification (crisp -> membership degrees)
│   ├── fuzzy_rules.py             # 27 Mamdani rule definitions
│   ├── fuzzy_system.py            # Custom NumPy Mamdani FIS Engine (14-gene parameter vector)
│   ├── defuzzification.py         # Centroid (COA), Bisector, MOM, SOM, LOM defuzzifiers
│   ├── sugeno_model.py            # Zero-Order Sugeno FIS model
│   ├── traffic_simulator.py       # Software intersection simulator (replaces hardware) + 10 scenarios
│   ├── controllers.py             # Fixed-time, Queue-proportional and FIS controllers
│   ├── genetic_optimizer.py       # Real-Coded GA optimizer for MF parameter tuning
│   └── utils.py                   # Green-time categories, parameter I/O & plotting helpers
│
├── app/
│   └── streamlit_app.py           # Streamlit Web Application Dashboard
│
├── results/
│   ├── membership_plots/          # Default & GA-tuned MF plots
│   ├── surface_plots/             # 2D control-surface heat-maps
│   ├── comparison_plots/          # Defuzzification curve, GA convergence, delay comparison, queue time-series
│   └── *.csv / ga_best_params.json  # Benchmark tables and tuned parameters
│
├── main.py                        # Diagnostic runner for test cases, GA & simulation benchmarks
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── PROJECT_REPORT.md              # Full project report
├── PPT_PRESENTATION.md            # Slide-by-slide presentation content
└── VIVA_QA_GUIDE.md               # Viva questions & answers
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone/navigate to directory and install requirements:

```
pip install -r requirements.txt
```

### 2. Run Diagnostic Suite & Benchmark

```
python main.py
```

This executes the 10 diagnostic test cases, benchmarks Mamdani vs Sugeno, runs the GA optimizer (about 2 minutes), simulates 10 scenarios x 5 controllers x 5 seeds, and updates all plots and tables under `results/`.

Faster options: `python main.py --quick` (small GA) or `python main.py --skip-ga` (re-use saved GA result).

### 3. Launch Streamlit Web GUI Dashboard

```
streamlit run app/streamlit_app.py
```

---

## 🚦 Green-Time Categories

| Green Time (s) | Category       | Action / Status                                         |
| -------------- | -------------- | ------------------------------------------------------- |
| **10 - 20**    | **Very Short** | Light demand - release the junction quickly.            |
| **20 - 30**    | **Short**      | Low-moderate demand - short phase keeps cycle compact.  |
| **30 - 40**    | **Medium**     | Balanced demand - standard phase length.                |
| **40 - 50**    | **Long**       | Heavy demand - extend green to clear the queue.         |
| **50 - 60**    | **Very Long**  | Saturated approach - maximum green allowed.             |

---

## 📊 Results Summary

Simulation benchmark: 10 traffic scenarios x 5 unseen random seeds, 1 simulated hour each (numbers from `results/overall_summary.csv`).

| Controller                 | Avg delay (s/veh) | vs Fixed-Time | Avg queue (veh) | Max queue (veh) |
| -------------------------- | ----------------- | ------------- | --------------- | --------------- |
| Fixed-Time (30 s / 30 s)   | 86.71             | -             | 52.99           | 109.78          |
| Queue-Proportional (crisp) | 32.68             | -62.3 %       | 18.86           | 43.14           |
| Fuzzy Mamdani (expert)     | 29.05             | -66.5 %       | 17.26           | 40.28           |
| Fuzzy Sugeno (expert)      | 27.74             | -68.0 %       | 16.48           | 38.70           |
| Fuzzy-GA Mamdani (tuned)   | 28.01             | -67.7 %       | 16.59           | 38.78           |

- All fuzzy controllers cut average delay by about two-thirds compared with a fixed 30 s / 30 s plan; the gap is largest on unbalanced demand (e.g. -88.5 % on the heavy-arterial scenarios).
- The GA lowered the **training** delay by 9.2 % (34.38 -> 31.20 s/veh) and improved the expert Mamdani FIS by 3.6 % on the full unseen-seed benchmark (better in 7 of 10 scenarios).
- The expert Sugeno FIS was marginally the best overall, and the crisp Queue-Proportional controller won the two most saturated scenarios (S03, S10). Full discussion in `PROJECT_REPORT.md`.

---

## 🎓 Academic Syllabus Alignment (Units 1 - 6)

- **Unit 1**: Soft Computing vs Hard Computing, real-world control application (urban traffic signals).
- **Unit 2**: Crisp sets vs Fuzzy sets, Triangular / Trapezoidal / Gaussian MFs, set operations (min / max).
- **Unit 3**: Fuzzification, Mamdani FIS, Sugeno FIS, Centroid Defuzzification.
- **Unit 4**: Genetic Algorithm fundamentals, real-coded encoding, fitness evaluation.
- **Unit 5**: Selection (Tournament), Crossover (Arithmetic), Mutation (Gaussian), Elitism, Convergence.
- **Unit 6**: Hybrid Fuzzy-GA intelligent optimization system.

---

## ⚠️ Scope & Limitations

Software-only study on a simplified single intersection (2 phases, no turning movements, pedestrians or emergency vehicles). Results come from simulated Poisson traffic, not field data. See `PROJECT_REPORT.md` for details and future scope (hardware / SUMO integration).
