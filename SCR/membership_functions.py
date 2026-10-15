"""
membership_functions.py
=======================
Membership function (MF) maths and small classes used by the fuzzy engine.

Supported shapes (Unit 2 - Fuzzy sets):
    * Triangular   : trimf(x, a, b, c)
    * Trapezoidal  : trapmf(x, a, b, c, d)
    * Gaussian     : gaussmf(x, mean, sigma)

Shoulder shapes (a == b or c == d) are handled explicitly so that the
"Low" and "High" sets at the edges of a universe saturate at 1.0.
"""
import numpy as np


def trimf(x, a, b, c):
    """Triangular MF with feet at a and c and peak at b."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else (x >= b).astype(float)
    right = (c - x) / (c - b) if c > b else (x <= b).astype(float)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapmf(x, a, b, c, d):
    """Trapezoidal MF: rises a->b, flat b->c, falls c->d."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (d - x) / (d - c) if d > c else (x <= d).astype(float)
    return np.clip(np.minimum(left, np.minimum(1.0, right)), 0.0, 1.0)


def gaussmf(x, mean, sigma):
    """Gaussian MF centred at `mean` with spread `sigma`."""
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((x - mean) / max(sigma, 1e-9)) ** 2)


class MembershipFunction:
    """Base class: a named fuzzy set (linguistic term) with a shape."""

    def __init__(self, name, params):
        self.name = name
        self.params = tuple(float(p) for p in params)

    def __call__(self, x):
        raise NotImplementedError

    @property
    def peak(self):
        """Representative crisp value of the set (used by Sugeno consequents)."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.params})"


class TriangularMF(MembershipFunction):
    def __call__(self, x):
        return trimf(x, *self.params)

    @property
    def peak(self):
        return self.params[1]


class TrapezoidalMF(MembershipFunction):
    def __call__(self, x):
        return trapmf(x, *self.params)

    @property
    def peak(self):
        return 0.5 * (self.params[1] + self.params[2])


class GaussianMF(MembershipFunction):
    def __call__(self, x):
        return gaussmf(x, *self.params)

    @property
    def peak(self):
        return self.params[0]
