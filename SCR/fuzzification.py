"""
fuzzification.py
================
Fuzzification: crisp sensor values -> membership degrees of each linguistic
term (Unit 3, step 1 of a fuzzy inference system).

A LinguisticVariable bundles a name, a universe of discourse and its fuzzy
terms (Low / Medium / High ...).
"""
import numpy as np


class LinguisticVariable:
    def __init__(self, name, universe, terms, unit=""):
        self.name = name
        self.universe = (float(universe[0]), float(universe[1]))
        self.terms = dict(terms)            # term name -> MembershipFunction
        self.unit = unit

    def clip(self, x):
        return float(np.clip(x, self.universe[0], self.universe[1]))

    def fuzzify(self, x):
        """Return {term: membership degree} for a crisp value x."""
        x = self.clip(x)
        return {t: float(mf(x)) for t, mf in self.terms.items()}

    def grid(self, n=401):
        return np.linspace(self.universe[0], self.universe[1], n)


def fuzzify_inputs(variables, crisp):
    """Fuzzify every input variable.  variables: {name: LinguisticVariable}."""
    return {name: var.fuzzify(crisp[name]) for name, var in variables.items()}
