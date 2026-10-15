"""
fuzzy_system.py
===============
Custom NumPy implementation of a Mamdani Fuzzy Inference System (FIS) for
traffic-signal green-time control (Unit 3).

Pipeline:  fuzzification -> rule firing (AND = min) -> implication (clip)
           -> aggregation (max) -> defuzzification (centroid)

The membership functions are described by a 14-gene parameter vector so the
Genetic Algorithm can tune them (Units 4-6):

    genes 0-2   Q_THIS   breakpoints  p1<p2<p3
    genes 3-5   Q_OTHER  breakpoints
    genes 6-8   ARRIVAL  breakpoints
    genes 9-13  GREEN    centres c1<c2<c3<c4<c5
"""
import numpy as np

from .membership_functions import TriangularMF, TrapezoidalMF
from .fuzzification import LinguisticVariable, fuzzify_inputs
from .defuzzification import DEFUZZIFIERS
from .fuzzy_rules import RULES, INPUT_NAMES

# ------------------------------------------------------------------ universes
Q_UNIVERSE = (0.0, 60.0)        # vehicles waiting on a 2-direction approach group
ARR_UNIVERSE = (0.0, 40.0)      # veh/min on a 2-direction approach group
GREEN_UNIVERSE = (10.0, 60.0)   # seconds

OUT_TERMS = ["VeryShort", "Short", "Medium", "Long", "VeryLong"]

# ------------------------------------------------------- expert (default) genes
DEFAULT_PARAMS = np.array([
    8, 22, 42,            # Q_THIS
    8, 22, 42,            # Q_OTHER
    8, 20, 32,            # ARRIVAL
    14, 24, 35, 46, 56,   # GREEN centres
], dtype=float)

GENE_NAMES = (["Q_THIS_p1", "Q_THIS_p2", "Q_THIS_p3",
               "Q_OTHER_p1", "Q_OTHER_p2", "Q_OTHER_p3",
               "ARR_p1", "ARR_p2", "ARR_p3"] +
              [f"GREEN_c{i}" for i in range(1, 6)])

# lower / upper bounds used by the GA
GENE_BOUNDS = np.array([
    (2, 20), (10, 40), (25, 58),
    (2, 20), (10, 40), (25, 58),
    (2, 15), (8, 28), (16, 38),
    (10, 22), (15, 32), (22, 42), (32, 54), (42, 60),
], dtype=float)

_GROUPS = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 14)]


def repair(params, min_gap=2.0):
    """Keep every group of breakpoints sorted, separated and inside bounds."""
    p = np.clip(np.asarray(params, dtype=float).copy(), GENE_BOUNDS[:, 0], GENE_BOUNDS[:, 1])
    for g in _GROUPS:
        v = np.sort(p[g])
        for i in range(1, len(v)):
            if v[i] - v[i - 1] < min_gap:
                v[i] = v[i - 1] + min_gap
        p[g] = np.minimum(v, GENE_BOUNDS[g, 1])
        p[g] = np.sort(p[g])
    return p


def _three_set_variable(name, universe, p, unit):
    lo, hi = universe
    p1, p2, p3 = p
    terms = {
        "Low": TrapezoidalMF("Low", (lo, lo, p1, p2)),
        "Medium": TriangularMF("Medium", (p1, p2, p3)),
        "High": TrapezoidalMF("High", (p2, p3, hi, hi)),
    }
    return LinguisticVariable(name, universe, terms, unit)


def _green_variable(c):
    lo, hi = GREEN_UNIVERSE
    c1, c2, c3, c4, c5 = c
    terms = {
        "VeryShort": TrapezoidalMF("VeryShort", (lo, lo, c1, c2)),
        "Short": TriangularMF("Short", (c1, c2, c3)),
        "Medium": TriangularMF("Medium", (c2, c3, c4)),
        "Long": TriangularMF("Long", (c3, c4, c5)),
        "VeryLong": TrapezoidalMF("VeryLong", (c4, c5, hi, hi)),
    }
    return LinguisticVariable("GREEN", GREEN_UNIVERSE, terms, "s")


def build_variables(params=None):
    p = repair(DEFAULT_PARAMS if params is None else params)
    inputs = {
        "Q_THIS": _three_set_variable("Q_THIS", Q_UNIVERSE, p[0:3], "veh"),
        "Q_OTHER": _three_set_variable("Q_OTHER", Q_UNIVERSE, p[3:6], "veh"),
        "ARRIVAL": _three_set_variable("ARRIVAL", ARR_UNIVERSE, p[6:9], "veh/min"),
    }
    return inputs, _green_variable(p[9:14])


class MamdaniFIS:
    """Mamdani FIS with min-AND, min-implication, max-aggregation."""

    def __init__(self, params=None, rules=None, defuzz="centroid", resolution=401):
        self.params = repair(DEFAULT_PARAMS if params is None else params)
        self.inputs, self.output = build_variables(self.params)
        self.rules = list(RULES if rules is None else rules)
        self.defuzz_name = defuzz
        self._defuzz = DEFUZZIFIERS[defuzz]
        self.y = self.output.grid(resolution)
        self._out_mu = {t: mf(self.y) for t, mf in self.output.terms.items()}

    # -------------------------------------------------------------- inference
    def infer(self, q_this, q_other, arrival, details=False):
        crisp = {"Q_THIS": q_this, "Q_OTHER": q_other, "ARRIVAL": arrival}
        fuzzified = fuzzify_inputs(self.inputs, crisp)

        strengths = []
        term_strength = {t: 0.0 for t in self.output.terms}
        for r in self.rules:
            w = min(fuzzified[v][r.antecedent[v]] for v in INPUT_NAMES)   # AND = min
            strengths.append(w)
            if w > term_strength[r.consequent]:                           # OR  = max
                term_strength[r.consequent] = w

        mu = np.zeros_like(self.y)
        for t, w in term_strength.items():
            if w > 0.0:
                mu = np.maximum(mu, np.minimum(w, self._out_mu[t]))       # clip + max
        green = self._defuzz(self.y, mu)

        if not details:
            return green
        return {"green": green, "fuzzified": fuzzified, "strengths": strengths,
                "term_strength": term_strength, "y": self.y, "mu": mu,
                "output_sets": self._out_mu}

    def infer_from_state(self, state):
        return self.infer(state["q_this"], state["q_other"], state["arrival_this"])


def build_traffic_fis(params=None, **kw):
    return MamdaniFIS(params=params, **kw)
