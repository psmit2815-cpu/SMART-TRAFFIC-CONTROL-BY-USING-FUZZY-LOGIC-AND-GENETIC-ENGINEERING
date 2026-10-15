"""
traffic_simulator.py
====================
Pure-software replacement for the hardware set-up (sensors + signal heads).

A 4-way intersection with two phases is simulated at 1-second resolution:

    Phase NS : North + South approaches get green together
    Phase EW : East  + West  approaches get green together

Virtual detectors report the queue length of each approach and the recent
arrival rate.  At the start of every phase the controller returns the green
duration.  After each green the signal spends YELLOW + ALL_RED seconds in
lost time (no discharge), then the other phase starts.

Vehicle arrivals are Poisson (rate may vary with time); departures follow a
saturation flow with a start-up lost time.  The key metric is the average
delay per vehicle (= total vehicle-seconds spent waiting / vehicles arrived).
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

DIRS = ("N", "S", "E", "W")
PHASE_DIRS = {"NS": (0, 1), "EW": (2, 3)}        # indices into DIRS


@dataclass
class SimConfig:
    duration: int = 3600          # simulated seconds
    sat_flow: float = 0.5         # veh/s per approach  (= 1800 veh/h)
    startup_lost: int = 2         # s of green before discharge begins
    yellow: int = 3
    all_red: int = 1
    min_green: float = 10.0
    max_green: float = 60.0
    detector_window: int = 30     # s window for arrival-rate estimate


@dataclass
class Scenario:
    name: str
    description: str
    rate_fn: object               # rate_fn(t) -> (N, S, E, W) in veh/min


def _const(n, s, e, w):
    return lambda t: (n, s, e, w)


def _ramp(base, peak, t0, t1, t2, t3):
    """Trapezoid profile: base -> peak (t0..t1), hold, peak -> base (t2..t3)."""
    def f(t):
        if t < t0 or t >= t3:
            return base
        if t < t1:
            return base + (peak - base) * (t - t0) / (t1 - t0)
        if t < t2:
            return peak
        return peak - (peak - base) * (t - t2) / (t3 - t2)
    return f


def _build_scenarios():
    sc = []
    sc.append(Scenario("S01_light_balanced", "Night / off-peak: 4 veh/min on every approach",
                       _const(4, 4, 4, 4)))
    sc.append(Scenario("S02_moderate_balanced", "Daytime: 8 veh/min on every approach",
                       _const(8, 8, 8, 8)))
    sc.append(Scenario("S03_heavy_balanced", "Busy junction: 12 veh/min on every approach",
                       _const(12, 12, 12, 12)))
    sc.append(Scenario("S04_ns_arterial_heavy", "Main road N-S heavy (14), side road E-W light (5)",
                       _const(14, 14, 5, 5)))
    sc.append(Scenario("S05_ew_arterial_heavy", "Main road E-W heavy (14), side road N-S light (5)",
                       _const(5, 5, 14, 14)))
    ns, ew = _ramp(6, 14, 300, 900, 1500, 2100), _const(6, 6, 6, 6)
    sc.append(Scenario("S06_morning_peak_ns", "N-S demand ramps 6 -> 14 -> 6 veh/min (peak), E-W steady 6",
                       lambda t: (ns(t), ns(t), 6, 6)))
    ew2 = _ramp(6, 14, 1500, 2100, 2700, 3300)
    sc.append(Scenario("S07_evening_peak_ew", "E-W demand ramps 6 -> 14 -> 6 veh/min late in the hour",
                       lambda t: (6, 6, ew2(t), ew2(t))))
    surge = _ramp(5, 13, 1000, 1200, 2200, 2400)
    sc.append(Scenario("S08_rush_hour_surge", "All approaches surge 5 -> 13 veh/min for ~20 minutes",
                       lambda t: (surge(t),) * 4))
    sc.append(Scenario("S09_fluctuating", "Sinusoidal demand, N-S and E-W out of phase (4..14 veh/min)",
                       lambda t: (9 + 5 * np.sin(2 * np.pi * t / 1200),
                                  9 + 5 * np.sin(2 * np.pi * t / 1200),
                                  9 - 5 * np.sin(2 * np.pi * t / 1200),
                                  9 - 5 * np.sin(2 * np.pi * t / 1200))))
    sc.append(Scenario("S10_one_sided_heavy", "Unbalanced: N 16, S 4, E 8, W 8 veh/min",
                       _const(16, 4, 8, 8)))
    return {s.name: s for s in sc}


SCENARIOS = _build_scenarios()


@dataclass
class SimResult:
    controller: str
    scenario: str
    seed: int
    avg_delay: float              # s / vehicle
    avg_queue: float              # vehicles (all 4 approaches)
    max_queue: int
    throughput: float             # vehicles discharged per hour
    arrived: int
    served: int
    final_queue: int
    avg_green: float
    n_phases: int
    history: dict                 # per-second logs (queues, phase, signal)
    phase_log: list               # (start_time, phase, green_time)


class TrafficSimulator:
    def __init__(self, scenario, controller, config=None, seed=0):
        self.scenario = scenario if isinstance(scenario, Scenario) else SCENARIOS[scenario]
        self.controller = controller
        self.cfg = config or SimConfig()
        self.seed = seed

    def _arrivals(self):
        rng = np.random.default_rng(self.seed)          # same seed -> same traffic for every controller
        t = np.arange(self.cfg.duration)
        rates = np.array([self.scenario.rate_fn(x) for x in t], dtype=float) / 60.0
        return rng.poisson(rates)

    def run(self, record=True):
        cfg = self.cfg
        arr = self._arrivals()
        T = cfg.duration
        q = [0, 0, 0, 0]
        credit = [0.0] * 4
        window = deque(maxlen=cfg.detector_window)       # per-second arrivals per direction
        lost = cfg.yellow + cfg.all_red

        total_wait = 0
        served = 0
        max_q = 0
        phases = ["NS", "EW"]
        phase_idx = 0
        phase_log = []
        greens = []

        hist = {"q": np.zeros((T, 4), dtype=np.int32), "phase": np.zeros(T, dtype=np.int8),
                "light": np.zeros(T, dtype=np.int8)} if record else None   # light: 0 red-all,1 NS green,2 EW green,3 yellow/all-red

        t = 0
        while t < T:
            ph = phases[phase_idx]
            idx = PHASE_DIRS[ph]
            other = PHASE_DIRS[phases[1 - phase_idx]]
            wsum = np.sum(window, axis=0) if window else np.zeros(4)
            scale = 60.0 / max(len(window), 1)
            state = {"q_this": q[idx[0]] + q[idx[1]],
                     "q_other": q[other[0]] + q[other[1]],
                     "arrival_this": float((wsum[idx[0]] + wsum[idx[1]]) * scale)}
            g = float(np.clip(self.controller.green_time(state), cfg.min_green, cfg.max_green))
            g_int = int(round(g))
            phase_log.append((t, ph, g_int))
            greens.append(g_int)

            # ------------------ green interval then lost time ------------------
            for k in range(g_int + lost):
                if t >= T:
                    break
                a = arr[t]
                for d in range(4):
                    q[d] += a[d]
                window.append(a)
                if k < g_int and k >= cfg.startup_lost:
                    for d in idx:
                        if q[d] > 0:
                            credit[d] += cfg.sat_flow
                            while credit[d] >= 1.0 and q[d] > 0:
                                q[d] -= 1
                                credit[d] -= 1.0
                                served += 1
                        else:
                            credit[d] = 0.0
                tot = q[0] + q[1] + q[2] + q[3]
                total_wait += tot
                if tot > max_q:
                    max_q = tot
                if record:
                    hist["q"][t] = q
                    hist["phase"][t] = phase_idx
                    hist["light"][t] = (1 + phase_idx) if k < g_int else 3
                t += 1
            credit = [0.0] * 4
            phase_idx = 1 - phase_idx

        arrived = int(arr.sum())
        final_q = q[0] + q[1] + q[2] + q[3]
        return SimResult(
            controller=self.controller.name, scenario=self.scenario.name, seed=self.seed,
            avg_delay=total_wait / max(arrived, 1), avg_queue=total_wait / T, max_queue=int(max_q),
            throughput=served * 3600.0 / T, arrived=arrived, served=served, final_queue=int(final_q),
            avg_green=float(np.mean(greens)), n_phases=len(greens),
            history=hist if record else {}, phase_log=phase_log)


def run_simulation(scenario, controller, seed=0, duration=3600, record=False):
    return TrafficSimulator(scenario, controller, SimConfig(duration=duration), seed).run(record=record)
