"""
controllers.py
==============
Signal controllers compared in the software simulation.  Each controller
exposes   green_time(state) -> seconds   and a `name`.

state = {"q_this", "q_other", "arrival_this"}   (see traffic_simulator.py)
"""
import numpy as np


class FixedTimeController:
    """Conventional pre-timed signal: same green every cycle."""

    def __init__(self, green=30.0):
        self.green = float(green)
        self.name = f"Fixed-Time ({int(green)}s)"

    def green_time(self, state):
        return self.green


class QueueProportionalController:
    """Crisp adaptive baseline: split a 60 s cycle in proportion to queues."""

    def __init__(self, cycle=60.0):
        self.cycle = cycle
        self.name = "Queue-Proportional (crisp)"

    def green_time(self, state):
        q1, q2 = state["q_this"], state["q_other"]
        return float(np.clip(self.cycle * (q1 + 1.0) / (q1 + q2 + 2.0), 10.0, 60.0))


class FISController:
    """Wraps any FIS object exposing infer_from_state()."""

    def __init__(self, fis, name):
        self.fis = fis
        self.name = name

    def green_time(self, state):
        return float(self.fis.infer_from_state(state))
