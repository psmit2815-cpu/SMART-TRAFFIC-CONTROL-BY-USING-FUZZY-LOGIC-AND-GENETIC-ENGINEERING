"""
defuzzification.py
==================
Defuzzifiers that convert an aggregated output fuzzy set (y, mu) into one
crisp green-time value (Unit 3).

    centroid  (COA) : centre of area               <- default, smoothest
    bisector  (BOA) : vertical line splitting area in two equal halves
    mom             : mean of maximum
    som / lom       : smallest / largest of maximum
"""
import numpy as np


def centroid(y, mu):
    area = np.sum(mu)
    if area <= 1e-12:
        return float(0.5 * (y[0] + y[-1]))          # safety net: mid-universe
    return float(np.sum(y * mu) / area)


def bisector(y, mu):
    c = np.cumsum(mu)
    if c[-1] <= 1e-12:
        return float(0.5 * (y[0] + y[-1]))
    return float(y[np.searchsorted(c, c[-1] / 2.0)])


def _maxima(y, mu):
    m = mu.max()
    return y[np.isclose(mu, m)] if m > 1e-12 else np.array([0.5 * (y[0] + y[-1])])


def mom(y, mu):
    return float(np.mean(_maxima(y, mu)))


def som(y, mu):
    return float(np.min(_maxima(y, mu)))


def lom(y, mu):
    return float(np.max(_maxima(y, mu)))


DEFUZZIFIERS = {"centroid": centroid, "bisector": bisector,
                "mom": mom, "som": som, "lom": lom}
