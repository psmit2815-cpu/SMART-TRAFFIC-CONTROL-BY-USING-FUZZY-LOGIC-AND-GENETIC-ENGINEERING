# Viva Q&A Guide - Smart Traffic Signal Control (Fuzzy Logic + GA)

Answers are written to match **this implementation** (numbers come from `results/`).

---

## A. Project Basics

**Q1. What is your project about?**
A fuzzy-logic controller that decides the green time of each traffic-signal phase from queue lengths and arrival rate, with a genetic algorithm tuning its membership functions. Everything runs in software using a traffic simulator.

**Q2. Why did you choose fuzzy logic for traffic control?**
Traffic is nonlinear and noisy, an exact model is hard, but operators reason in words ("long queue, high arrivals -> long green"). Fuzzy logic turns such rules directly into a controller.

**Q3. Why software-only? How did you replace the hardware?**
Detectors are virtual (queue per approach and a 30 s arrival-rate window), the controller is a Python function, the signal heads are a simulated state machine (green -> yellow -> all-red), and traffic is Poisson arrivals. This allows repeatable tests on 10 scenarios.

**Q4. What is the difference between hard computing and soft computing?**
Hard computing needs precise models and gives exact answers. Soft computing (fuzzy logic, GA, neural networks) tolerates imprecision and uncertainty and gives approximate but practical solutions.

**Q5. What are the inputs and output?**
Inputs: Q_THIS (queue on approach about to get green, 0-60 veh), Q_OTHER (opposing queue, 0-60 veh), ARRIVAL (veh/min, 0-40). Output: GREEN time, 10-60 s.

---

## B. Fuzzy Logic

**Q6. What is a fuzzy set? How is it different from a crisp set?**
A crisp set has membership 0 or 1; a fuzzy set allows any degree in [0, 1]. A queue of 22 vehicles can be "Medium" to degree 1.0 and partly "Low" or "High" near the borders.

**Q7. Which membership functions did you use?**
Triangular (Medium and mid output sets) and trapezoidal (Low / High shoulders, VeryShort / VeryLong). Gaussian is also implemented in `membership_functions.py` for comparison.

**Q8. Why shoulder (saturating) sets at the edges?**
So that a queue of 0 is fully "Low" and a queue of 60 is fully "High"; otherwise membership would fall to zero at the ends of the universe.

**Q9. Explain the Mamdani inference steps.**
Fuzzification -> rule firing with min (AND) -> implication by clipping output sets -> aggregation with max -> centroid defuzzification.

**Q10. Why centroid defuzzification?**
It uses the whole aggregated shape, gives smooth output as inputs change, and every active rule influences the result. MOM, bisector, SOM and LOM are also implemented.

**Q11. How many rules? How did you build them?**
27 = 3 x 3 x 3 (all combinations of Low / Medium / High for the three inputs). Principle: more queue and arrivals on the green side -> longer green; a long opposing queue -> shorter green.

**Q12. What if no rule fires?**
With full 3-term partitions every input value activates at least one term per input, so at least one rule fires. As a safety net the code still returns the middle of the output universe if the aggregated area is zero.

**Q13. Mamdani vs Sugeno?**
Mamdani consequents are fuzzy sets (interpretable, needs defuzzification); zero-order Sugeno consequents are constants and the output is a weighted average (cheaper, easier to analyse). In my tests their outputs correlate at 0.988 and average 1.6 s apart.

**Q14. Which was faster?**
Sugeno, about 94 µs against about 114 µs per inference on my machine. Both are far below the 1-second control step.

**Q15. How does a long opposing queue affect the output?**
Rules with Q_OTHER = High give shorter greens when Q_THIS is Low or Medium (e.g. Low/High/any -> VeryShort), which avoids starving the waiting approach.

**Q16. Show one example calculation.**
Q_THIS = 30, Q_OTHER = 12, ARRIVAL = 20: fuzzify, fire the rules (e.g. Medium/Low/Medium -> Medium, High/Low/Medium -> Long), clip and aggregate, centroid. The dashboard "Signal Controller" tab shows the activated rules and the centroid curve live.

---

## C. Genetic Algorithm

**Q17. Why use a GA?**
Hand-chosen membership functions are rarely optimal. The GA searches the 14-dimensional parameter space using the simulator-measured delay as fitness; the objective is non-differentiable, which suits a GA.

**Q18. What is the chromosome?**
14 real numbers: three breakpoints for each of Q_THIS, Q_OTHER and ARRIVAL, plus five output centres for GREEN.

**Q19. What is the fitness function?**
Mean average vehicle delay (s/veh) over 7 training scenarios x 2 seeds (1800 s each). It is minimised.

**Q20. Which GA operators did you use?**
Tournament selection (k = 3), arithmetic crossover (probability 0.9), Gaussian mutation (rate 0.15, σ = 8 % of range), elitism of 2. A repair step keeps breakpoints sorted and inside bounds.

**Q21. Why real-coded instead of binary?**
The parameters are continuous, so real coding avoids encoding / decoding and gives finer control with arithmetic crossover.

**Q22. How do you avoid over-fitting?**
Training and test random seeds differ, and three scenarios (S01, S05, S07) are never used in training. Out of sample the GA gain was smaller (3.6 %) than on the training data (9.2 %), which I report honestly.

**Q23. What was the GA outcome?**
Training delay fell from 34.38 to 31.20 s/veh (-9.2 %); on the full 10-scenario test the tuned Mamdani FIS gave 28.01 s/veh against 29.05 s/veh for the expert design (-3.6 %), better in 7 of 10 scenarios.

**Q24. Why is the tuned FIS slightly worse in some scenarios?**
In S01, S06 and S09 it was 1-2 % worse. The fitness is an average dominated by heavy-traffic scenarios, so light traffic was not protected. A normalised or weighted fitness would help.

**Q25. What is elitism and why use it?**
Copying the best individuals unchanged into the next generation so the best solution found is never lost; the convergence curve is therefore monotonic.

**Q26. Did you seed the GA with the expert solution?**
Yes, one of the 24 initial individuals is the expert design, so the GA can never end up worse than it on the training objective. The other 23 are random.

---

## D. Simulation and Results

**Q27. How does the simulator work?**
1-second steps; Poisson arrivals per approach; two phases (N+S, E+W); each green discharges at 0.5 veh/s per approach after a 2 s start-up loss; 3 s yellow + 1 s all-red after each green; green clamped to 10-60 s.

**Q28. How is delay measured?**
Total vehicle-seconds spent waiting in queues divided by the number of vehicles that arrived.

**Q29. Why is the comparison fair?**
For a given seed every controller sees exactly the same arrivals (common random numbers), and tests use seeds the GA never saw.

**Q30. What are the main results?**
Average delay over all scenarios and seeds: Fixed 86.71 s, Queue-Proportional 32.68 s, Fuzzy Mamdani 29.05 s, Fuzzy Sugeno 27.74 s, Fuzzy-GA 28.01 s. Fuzzy controllers are about 66-68 % lower than fixed-time.

**Q31. Where is the improvement largest?**
On unbalanced demand: -88.5 % on the arterial-heavy scenarios S04 / S05, because a fixed 50/50 split starves the busy road.

**Q32. Is fuzzy always the best?**
No. The crisp Queue-Proportional controller won the two most saturated scenarios (S03, S10), and the expert Sugeno FIS was marginally best overall. Near capacity the delay is limited by the intersection's capacity itself.

**Q33. Why is throughput almost the same for all adaptive controllers?**
Throughput is limited by demand when the junction is not oversaturated (about 1995 veh/h); the benefit shows up as shorter queues and delays.

**Q34. Is the fixed-time baseline fair?**
It is a generic 30 s / 30 s plan. A plan optimised for one scenario (Webster method) would do better there, but it cannot adapt when demand changes.

**Q35. What does the 3,000-record dataset represent?**
Synthetic intersection states with a crisp queue-clearance reference green time. It checks that the FIS follows a sensible trend (correlation 0.945 for Mamdani, 0.957 for Sugeno); it is not field ground truth.

---

## E. Limitations and Extensions

**Q36. Limitations?**
Single intersection, two phases, no turning movements, pedestrians or emergency vehicles, Poisson arrivals, ideal detectors, simulated rather than field data.

**Q37. How would you deploy it on real hardware?**
The controller is a pure function of three inputs, so it can run on a microcontroller fed by loop or camera detectors and driving signal relays; safety interlocks (conflict monitor, minimum clearance times) must be added.

**Q38. How could it be improved?**
SUMO / CityFlow for realistic networks, multi-intersection coordination, type-2 fuzzy sets for sensor noise, multi-objective GA (delay, stops, fairness), phase sequencing and emergency-vehicle priority.

**Q39. What is the time complexity of one decision?**
O(R) with R = 27 rules plus a fixed 401-point defuzzification grid, about 0.1 ms, negligible against the 1 s control step.

**Q40. What is your key contribution?**
A complete hardware-free Fuzzy-GA traffic-signal study: custom Mamdani and Sugeno engines, a simulator with 10 scenarios, simulator-in-the-loop GA tuning, honest benchmarking against fixed-time and crisp baselines, and an interactive dashboard.
