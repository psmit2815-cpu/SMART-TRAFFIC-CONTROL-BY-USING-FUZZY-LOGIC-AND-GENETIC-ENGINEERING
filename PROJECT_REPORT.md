# Smart Traffic Signal Control Using Fuzzy Logic and Genetic Algorithms

**Semester 5 Electronics & Instrumentation Engineering - Soft Computing Final Project (Software-Only Implementation)**

| | |
|---|---|
| Student | PATEL SMIT KALPESH |
| Enrollment No. | 240170147018 |
| Department / Semester | Electronics and Instrumentation Engineering, Semester 5 |
| College | VGEC |

---

## Abstract

Fixed-time traffic signals repeat the same green split regardless of demand, so queues grow on busy approaches while green time is wasted on empty ones. This project designs a **fuzzy-logic traffic-signal controller** that decides the green duration of every phase from three detector measurements: the queue on the approach about to receive green, the queue on the opposing approach, and the arrival rate. A 27-rule **Mamdani Fuzzy Inference System** is implemented from scratch in NumPy and benchmarked against a **zero-order Sugeno FIS**. A **real-coded Genetic Algorithm** then tunes 14 membership-function parameters by minimising the average vehicle delay. The project is **entirely software-based**: a Python intersection simulator with virtual detectors replaces all hardware. Across 10 traffic scenarios and 5 unseen random seeds, the fuzzy controllers reduce average delay by 66.5 % - 68.0 % compared with a fixed 30 s / 30 s plan. The GA improved the training delay by 9.2 % and the expert Mamdani FIS by 3.6 % on the full benchmark. A Streamlit dashboard provides interactive control, simulation and comparison.

**Keywords:** fuzzy logic, Mamdani, Sugeno, genetic algorithm, traffic signal control, soft computing, simulation.

---

## 1. Introduction

Urban traffic congestion wastes time and fuel. Most junctions still use **pre-timed** signals whose plans are designed for an "average" demand. Real demand changes by time of day, direction and day-to-day randomness. Traffic control is also a task where precise mathematical models are hard to obtain, but human operators reason well with linguistic rules such as *"if the queue is long and arrivals are high, give a long green"*. This is exactly the situation where **soft computing** (fuzzy logic) is appropriate (Pappis & Mamdani, 1977).

## 2. Problem Statement

Design an adaptive controller that selects the green time of each phase at a two-phase intersection so as to reduce vehicle delay and queue length, **without hardware**, and optimise its parameters automatically.

## 3. Objectives

1. Implement a Mamdani FIS (fuzzification, rule base, inference, centroid defuzzification) without using a fuzzy toolbox.
2. Compare it with a zero-order Sugeno FIS.
3. Build a software traffic simulator that replaces sensors and signal hardware.
4. Optimise the membership functions with a real-coded Genetic Algorithm.
5. Benchmark against a fixed-time controller and a crisp adaptive controller.
6. Provide an interactive dashboard for demonstration.

## 4. Why Software-Only?

| Hardware version | Software version (this project) |
|---|---|
| Loop / camera detectors | Virtual detectors inside the simulator (queue length per approach, 30 s arrival-rate window) |
| Microcontroller running the controller | Python functions `green_time(state)` |
| Signal heads (R/Y/G LEDs) | Simulated signal state machine (green -> yellow -> all-red) and animated intersection drawing |
| Real traffic | Poisson arrivals with time-varying rates (10 scenarios) |
| Field measurement of delay | Vehicle-seconds of waiting accumulated every second |

## 5. System Architecture

```
 Poisson arrivals ──► Queues (N,S,E,W) ──► Virtual detectors
                                              │  Q_THIS, Q_OTHER, ARRIVAL
                                              ▼
                    ┌──────────── Controller ────────────┐
                    │ Fuzzification → 27 rules → Mamdani │ → green time (10-60 s)
                    │ aggregation → centroid defuzz.     │
                    └────────────────────────────────────┘
                                              │
                    Signal state machine ◄────┘   (green → yellow+all-red → next phase)
                                              │
                    Metrics: delay, queue, throughput        ◄── GA fitness (minimise delay)
```

The decision is taken **at the start of each phase** for the approach group (N+S or E+W) that is about to receive green.

## 6. Fuzzy Controller Design

### 6.1 Variables and Universes of Discourse

| Variable | Type | Universe | Terms |
|---|---|---|---|
| Q_THIS | input | 0 - 60 vehicles | Low, Medium, High |
| Q_OTHER | input | 0 - 60 vehicles | Low, Medium, High |
| ARRIVAL | input | 0 - 40 veh/min | Low, Medium, High |
| GREEN | output | 10 - 60 s | VeryShort, Short, Medium, Long, VeryLong |

Inputs use a Low (left-shoulder trapezoid), Medium (triangle) and High (right-shoulder trapezoid) partition defined by three breakpoints `p1 < p2 < p3`: Low = trap(min, min, p1, p2), Medium = tri(p1, p2, p3), High = trap(p2, p3, max, max). The output sets are defined by five centres `c1 < ... < c5`. Expert (default) breakpoints are Q: (8, 22, 42), ARRIVAL: (8, 20, 32), GREEN centres: (14, 24, 35, 46, 56).

### 6.2 Rule Base (27 rules)

Every combination of (Q_THIS, Q_OTHER, ARRIVAL) has a rule. Design principles: longer green when the green approach has a long queue and high arrivals; shorter green when the opposing approach is waiting with a long queue (fairness); very short green when the green approach is nearly empty.

| Q_THIS | Q_OTHER | ARRIVAL = Low | Medium | High |
|---|---|---|---|---|
| Low | Low | VeryShort | Short | Short |
| Low | Medium | VeryShort | VeryShort | Short |
| Low | High | VeryShort | VeryShort | VeryShort |
| Medium | Low | Short | Medium | Long |
| Medium | Medium | Short | Medium | Medium |
| Medium | High | VeryShort | Short | Medium |
| High | Low | Medium | Long | VeryLong |
| High | Medium | Medium | Long | Long |
| High | High | Medium | Medium | Long |

### 6.3 Inference Mechanism (Mamdani)

1. **Fuzzification** - membership degree of each crisp input in each term.
2. **Rule firing** - AND = `min` of the three antecedent degrees.
3. **Implication** - each output set is clipped at the firing strength (`min`).
4. **Aggregation** - clipped sets combined with `max`.
5. **Defuzzification** - centroid of area: `z = Σ y·μ(y) / Σ μ(y)` over a 401-point grid. If no rule fires, the mid-universe value is returned as a safety net.

### 6.4 Sugeno Benchmark

Same fuzzification and rules, but each consequent is a constant (the output centres `c1..c5`) and `z = Σ w_i z_i / Σ w_i`. It needs no output aggregation and is cheaper to compute.

### 6.5 Worked Example

Q_THIS = 30, Q_OTHER = 12, ARRIVAL = 20: Q_THIS is partly Medium and High, Q_OTHER is partly Low and Medium, ARRIVAL is mostly Medium. The rules "Medium/Low/Medium -> Medium", "High/Low/Medium -> Long" and neighbours fire with different strengths; the clipped Medium and Long sets are aggregated and the centroid is read off (see `results/comparison_plots/defuzzification_example.png` and the dashboard).

## 7. Software Simulation Environment

- **Intersection:** 4 approaches (N, S, E, W), 2 phases (NS, EW), one lane-equivalent per approach, saturation flow 0.5 veh/s (1800 veh/h).
- **Time step:** 1 s. **Duration:** 3600 s (benchmark) or 1800 s (GA training).
- **Arrivals:** Poisson with rate λ(t) per approach.
- **Departures:** discharge at saturation flow after a 2 s start-up lost time.
- **Lost time:** 3 s yellow + 1 s all-red after every green (no discharge).
- **Green limits:** 10 s minimum, 60 s maximum.
- **Detectors:** queue per approach; arrival rate from a sliding 30 s window.
- **Metrics:** average delay per vehicle = total vehicle-seconds waiting / vehicles arrived; average and maximum queue; throughput (veh/h); final queue.
- **Fair comparison:** for a given seed every controller sees exactly the same arrivals (common random numbers).

### Scenarios

| ID | Description |
|---|---|
| S01 | Light balanced (4 veh/min each approach) |
| S02 | Moderate balanced (8) |
| S03 | Heavy balanced (12) |
| S04 | N-S arterial heavy (14), E-W light (5) |
| S05 | E-W arterial heavy (14), N-S light (5) |
| S06 | Morning peak: N-S ramps 6 -> 14 -> 6, E-W steady 6 |
| S07 | Evening peak: E-W ramp late in the hour |
| S08 | Rush-hour surge on all approaches (5 -> 13 -> 5) |
| S09 | Sinusoidal, N-S and E-W out of phase |
| S10 | One-sided heavy: N 16, S 4, E 8, W 8 |

## 8. Genetic Algorithm Optimisation

| Item | Choice |
|---|---|
| Encoding | Real-valued chromosome, 14 genes (9 input breakpoints + 5 output centres) |
| Fitness (minimised) | Mean average vehicle delay over 7 training scenarios x 2 seeds, 1800 s each |
| Initial population | 24 individuals: 1 expert design + 23 random (repaired) |
| Selection | Tournament, k = 3 |
| Crossover | Arithmetic (blend), probability 0.9 |
| Mutation | Gaussian, per-gene probability 0.15, σ = 8 % of gene range |
| Elitism | 2 best individuals copied unchanged |
| Repair | Breakpoints sorted, minimum gap 2, clipped to bounds so the MFs stay valid |
| Generations | 30 (684 simulator-based fitness evaluations, about 116 s) |

**Generalisation protocol.** Training uses scenarios S02, S03, S04, S06, S08, S09, S10 with seeds 101 and 102. Testing uses all 10 scenarios with seeds 1-5, so S01, S05 and S07 are **unseen scenarios** and every test run uses unseen random traffic.

## 9. Results

### 9.1 Diagnostic Test Cases (expert FIS)

| Case | Q_THIS | Q_OTHER | ARRIVAL | Mamdani (s) | Sugeno (s) | Category |
|---|---|---|---|---|---|---|
| Empty junction | 0 | 0 | 0 | 14.9 | 14.0 | Very Short |
| Light traffic both sides | 5 | 5 | 6 | 14.9 | 14.0 | Very Short |
| Moderate balanced | 18 | 18 | 16 | 29.4 | 24.7 | Short |
| Heavy green side, light opposing | 45 | 8 | 30 | 52.4 | 54.3 | Very Long |
| Light green side, heavy opposing | 6 | 48 | 8 | 14.9 | 14.0 | Very Short |
| Heavy both sides | 52 | 52 | 34 | 45.7 | 46.0 | Long |
| Medium queue, high arrivals | 25 | 15 | 32 | 41.0 | 42.9 | Long |
| Large queue, low arrivals | 42 | 20 | 6 | 35.0 | 35.0 | Medium |
| Small queue, high arrivals | 8 | 10 | 30 | 23.6 | 22.9 | Short |
| Everything medium | 25 | 25 | 20 | 35.0 | 35.0 | Medium |

The outputs follow the intended behaviour: longer green for long queues / high arrivals, minimum-length green for empty approaches or when the opposing queue is long.

### 9.2 Dataset Benchmark (3,000 synthetic states)

The reference is a crisp queue-clearance heuristic (not field ground truth), so these numbers only show that the FIS follows a sensible trend.

| | MAE (s) | RMSE (s) | Correlation | Time / inference |
|---|---|---|---|---|
| Mamdani vs reference | 6.94 | 8.42 | 0.945 | about 114 µs |
| Sugeno vs reference | 5.89 | 7.68 | 0.957 | about 94 µs |
| Mamdani vs Sugeno | 1.60 | 2.18 | 0.988 | - |

### 9.3 GA Optimisation

Training delay fell from **34.38 to 31.20 s/veh (-9.2 %)**. Tuned parameters (full table in `results/ga_parameters.csv`) shift the GREEN centres and widen / narrow the queue and arrival sets; see `results/membership_plots/membership_ga_tuned.png` and `results/comparison_plots/ga_convergence.png`.

### 9.4 Simulation Benchmark - Average Delay (s/veh, mean of 5 unseen seeds)

| Scenario | Fixed 30 s | Queue-Prop. | Mamdani | Sugeno | Fuzzy-GA |
|---|---|---|---|---|---|
| S01 light balanced (unseen) | 15.6 | 21.7 | 12.3 | 12.2 | 12.6 |
| S02 moderate balanced | 19.0 | 25.9 | 18.2 | 17.4 | 17.3 |
| S03 heavy balanced | 60.5 | 39.7 | 41.9 | 41.1 | 40.4 |
| S04 N-S arterial heavy | 193.6 | 30.7 | 23.0 | 22.2 | 22.3 |
| S05 E-W arterial heavy (unseen) | 192.7 | 30.6 | 23.0 | 21.9 | 22.2 |
| S06 morning peak | 45.2 | 26.9 | 18.7 | 18.2 | 18.9 |
| S07 evening peak (unseen) | 36.8 | 26.6 | 18.5 | 17.8 | 18.0 |
| S08 rush-hour surge | 48.1 | 42.6 | 38.2 | 36.1 | 37.8 |
| S09 fluctuating | 39.1 | 28.2 | 20.6 | 20.1 | 21.0 |
| S10 one-sided heavy | 216.5 | 53.8 | 76.1 | 70.5 | 69.5 |

### 9.5 Overall Comparison

| Controller | Avg delay (s/veh) | vs Fixed | Avg queue | Max queue | Throughput (veh/h) | Scenarios won |
|---|---|---|---|---|---|---|
| Fixed-Time (30 s) | 86.71 | - | 52.99 | 109.78 | 1931.8 | 0 |
| Queue-Proportional | 32.68 | -62.3 % | 18.86 | 43.14 | 1994.4 | 2 |
| Fuzzy Mamdani (expert) | 29.05 | -66.5 % | 17.26 | 40.28 | 1994.8 | 0 |
| Fuzzy Sugeno (expert) | 27.74 | -68.0 % | 16.48 | 38.70 | 1995.0 | 7 |
| Fuzzy-GA Mamdani | 28.01 | -67.7 % | 16.59 | 38.78 | 1995.5 | 1 |

Mean delay on the three unseen scenarios (S01, S05, S07): Fixed 81.71, Queue-Prop. 26.33, Mamdani 17.95, Sugeno 17.29, Fuzzy-GA 17.60.

## 10. Discussion

1. **Fuzzy control clearly beats fixed-time.** The largest gains occur on unbalanced demand (S04/S05: -88.5 %, S10: -67.9 % for Fuzzy-GA) because the fixed plan gives half the cycle to the light direction and the heavy approach saturates. Even at light demand (S01) the fuzzy controller shortens the cycle and cuts delay by 19 %.
2. **Throughput is similar for all adaptive controllers** (about 1995 veh/h); the benefit shows up as much smaller queues and delays, and a smaller final queue (15.8 - 16.6 veh vs 79.5 for fixed-time).
3. **GA helps, but modestly out of sample.** The training gain (9.2 %) shrinks to 3.6 % on the full benchmark and the tuned FIS is slightly worse than the expert FIS in S01, S06 and S09 (+2.2 %, +1.1 %, +1.7 %). The GA optimised the mean delay, which is dominated by heavy scenarios, so light-traffic performance was not explicitly protected. A weighted or normalised fitness would address this.
4. **Sugeno was marginally best overall.** The expert Sugeno FIS tends to produce slightly shorter greens (e.g. 24.7 s vs 29.4 s for the moderate-balanced case), which suits this simulator. A fair statement is that Mamdani and Sugeno behave very similarly (output correlation 0.988); Sugeno is cheaper to compute and Mamdani has the more interpretable output sets.
5. **A crisp adaptive baseline is competitive on saturated demand.** The queue-proportional controller won S03 and S10, showing that the fuzzy controllers had their largest advantage over the crisp baseline in light-to-moderate and time-varying scenarios. S10 is close to capacity (the north approach alone needs about 53 % of the time as green, 16 veh/min against a 30 veh/min saturation flow, and E-W about 27 %, before lost time is added), so delay there is largely capacity-limited.
6. **Fixed-time baseline caveat.** The 30 s / 30 s plan is a generic plan. A plan optimised per scenario (e.g. with Webster's method) would perform better than the figures shown, but could not adapt to changing demand.

## 11. Limitations

- Single intersection, two phases, no turning movements, pedestrians, emergency vehicles or lane-level behaviour.
- Poisson arrivals and idealised detectors (no sensor noise or failure).
- The 3,000-record dataset uses a heuristic reference, not measured data.
- Delay counts waiting time in queues (no acceleration / deceleration model).
- GA results depend on the training scenarios and random seeds.

## 12. Future Scope

- Replace the built-in simulator with SUMO / CityFlow for realistic multi-lane networks.
- Add phase sequencing, pedestrian and emergency-vehicle priority, and multi-intersection coordination.
- Multi-objective GA (delay, stops, fairness) and type-2 fuzzy sets for sensor noise.
- Hardware deployment on a microcontroller with real detectors (the controller code is already a pure function of three inputs).

## 13. Conclusion

A fuzzy logic controller tuned by a genetic algorithm can adapt green time to live traffic using only three intuitive measurements. In a purely software study, fuzzy controllers reduced average delay by roughly two-thirds versus a fixed-time plan, with the largest improvements on unbalanced and time-varying demand, and the GA gave a further modest gain over the hand-designed Mamdani FIS. The complete pipeline - rule base, inference engine, optimiser, simulator and dashboard - runs without any hardware.

## 14. References

1. L. A. Zadeh, "Fuzzy sets," *Information and Control*, 8(3), 338-353, 1965.
2. E. H. Mamdani and S. Assilian, "An experiment in linguistic synthesis with a fuzzy logic controller," *Int. J. Man-Machine Studies*, 7(1), 1-13, 1975.
3. T. Takagi and M. Sugeno, "Fuzzy identification of systems and its applications to modeling and control," *IEEE Trans. Systems, Man, and Cybernetics*, 15(1), 116-132, 1985.
4. C. P. Pappis and E. H. Mamdani, "A fuzzy logic controller for a traffic junction," *IEEE Trans. Systems, Man, and Cybernetics*, 7(10), 707-717, 1977.
5. J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975.
6. F. V. Webster, "Traffic signal settings," Road Research Technical Paper No. 39, HMSO, London, 1958.
