# PPT Presentation - Smart Traffic Signal Control Using Fuzzy Logic and Genetic Algorithms

> Slide-by-slide content (title, bullets, speaker notes, suggested visual). Suggested length: 15 slides, about 12 minutes. Visuals are in `results/`.

---

## Slide 1 - Title
**Smart Traffic Signal Control Using Fuzzy Logic and Genetic Algorithms**
Soft Computing Final Project (Software-Only)
PATEL SMIT KALPESH | Enrollment: 240170147018 | Electronics and Instrumentation, Sem 5 | VGEC

---

## Slide 2 - Problem Statement
- Fixed-time signals ignore real traffic: same green every cycle
- Result: long queues on busy roads, wasted green on empty roads
- Traffic is hard to model exactly, but easy to describe in words: "long queue + many arrivals -> long green"
- **Goal:** adaptive green-time controller, optimised automatically, with **no hardware**

*Notes:* Motivate with a real junction everyone knows, one road clearly busier than the other.

---

## Slide 3 - Why Soft Computing?
- Hard computing needs an exact model; traffic is noisy and nonlinear
- Fuzzy logic: human-like IF-THEN rules, smooth transitions, no exact model
- Genetic algorithm: automatic tuning when we do not know the best membership functions
- Hybrid **Fuzzy + GA** = interpretable controller + data-driven optimisation

---

## Slide 4 - Software-Only Approach
| Hardware | Replaced by |
|---|---|
| Loop / camera detectors | Virtual detectors (queues, 30 s arrival window) |
| Microcontroller | Python controller function |
| Signal heads | Simulated signal state machine + animated view |
| Real traffic | Poisson arrivals, 10 scenarios |

*Visual:* intersection replay from the Streamlit "Intersection Simulation" tab.

---

## Slide 5 - System Architecture
Arrivals -> Queues -> Virtual detectors -> **Fuzzy controller** -> Green time -> Signal state machine -> Metrics (delay, queue, throughput) -> **GA fitness**

*Notes:* Decision is taken at the start of every phase for the approach group about to get green.

---

## Slide 6 - Fuzzy Controller: Inputs and Output
- **Q_THIS** (0-60 veh): queue on approach about to get green
- **Q_OTHER** (0-60 veh): queue on opposing approach
- **ARRIVAL** (0-40 veh/min): arrival rate on green approach
- **GREEN** (10-60 s): VeryShort, Short, Medium, Long, VeryLong
- Input terms: Low / Medium / High

*Visual:* `results/membership_plots/membership_default.png`

---

## Slide 7 - Rule Base (27 rules)
- Full 3 x 3 x 3 grid
- Examples:
  - IF Q_THIS is High AND Q_OTHER is Low AND ARRIVAL is High THEN GREEN is VeryLong
  - IF Q_THIS is Low AND Q_OTHER is High THEN GREEN is VeryShort
  - IF Q_THIS is Medium AND Q_OTHER is Medium AND ARRIVAL is Medium THEN GREEN is Medium
- Fairness built in: long opposing queue shortens the green

---

## Slide 8 - Mamdani Inference Steps
1. Fuzzification
2. Rule firing (AND = min)
3. Implication (clip)
4. Aggregation (max)
5. Centroid defuzzification

*Visual:* `results/comparison_plots/defuzzification_example.png` (Q_THIS=30, Q_OTHER=12, ARRIVAL=20)

---

## Slide 9 - Mamdani vs Sugeno
- Same inputs and same 27 rules
- Mamdani: fuzzy output sets + centroid (interpretable, smoother)
- Sugeno (zero-order): constant outputs + weighted average (cheaper)
- Output agreement on 3,000 states: correlation **0.988**, MAE **1.60 s**

---

## Slide 10 - Genetic Algorithm Tuning
- Chromosome: 14 real genes (9 input breakpoints + 5 output centres)
- Fitness: mean vehicle delay from the simulator (minimise)
- Tournament selection, arithmetic crossover, Gaussian mutation, elitism
- Repair step keeps membership functions valid
- 24 individuals x 30 generations = 684 evaluations (about 2 min)

*Visual:* `results/comparison_plots/ga_convergence.png`

---

## Slide 11 - Simulation Setup
- 4-way, 2-phase intersection, 1 s time step, 1 hour per run
- Saturation flow 1800 veh/h per approach, yellow 3 s + all-red 1 s
- 10 scenarios: light, moderate, heavy, arterial-heavy (x2), morning/evening peak, surge, fluctuating, one-sided
- Tested on 5 unseen random seeds; 3 scenarios unseen by the GA
- Same traffic for every controller (fair comparison)

---

## Slide 12 - Results: Delay
| Controller | Avg delay (s/veh) | vs Fixed |
|---|---|---|
| Fixed-Time 30/30 | 86.71 | - |
| Queue-Proportional (crisp) | 32.68 | -62.3 % |
| Fuzzy Mamdani | 29.05 | -66.5 % |
| Fuzzy Sugeno | 27.74 | -68.0 % |
| Fuzzy-GA Mamdani | 28.01 | -67.7 % |

*Visual:* `results/comparison_plots/delay_comparison.png`

---

## Slide 13 - Results: Queues and Behaviour
- Average queue: 52.99 (fixed) -> about 16.5-17.3 veh (fuzzy)
- Maximum queue: 109.78 -> about 38.7-40.3 veh
- Throughput similar (about 1995 veh/h): benefit is shorter queues and waits
- Biggest gain on unbalanced demand: **-88.5 %** on the arterial-heavy scenarios

*Visual:* `results/comparison_plots/queue_timeseries_S06.png`

---

## Slide 14 - Honest Findings
- GA: training delay **-9.2 %** (34.38 -> 31.20), full benchmark **-3.6 %** vs expert Mamdani; better in 7/10 scenarios
- Slightly worse in light / fluctuating traffic (S01, S06, S09): fitness favoured heavy scenarios
- Expert Sugeno marginally best overall; crisp controller won the two most saturated scenarios (S03, S10)
- Limits: single junction, simplified traffic model

---

## Slide 15 - Dashboard Demo
- Live green-time calculator with activated rules
- Control-surface heat-maps
- Animated intersection replay
- Controller comparison and in-app GA run

*Notes:* `streamlit run app/streamlit_app.py`. Demo one heavy-vs-light case and show fixed-time vs fuzzy.

---

## Slide 16 - Conclusion and Future Scope
- Fuzzy logic gives an interpretable adaptive controller that cut delay by about two-thirds vs fixed-time
- GA tuning adds a further modest gain
- Future: SUMO / CityFlow, multiple intersections, pedestrian / emergency priority, multi-objective GA, hardware deployment

**Thank you - Questions?**
