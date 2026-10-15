"""
fuzzy_rules.py
==============
27 Mamdani rules for the green-time decision.

Decision is taken at the START of every phase, for the approach group that is
about to receive green:

    Q_THIS  : vehicles queued on the approach group about to get GREEN
    Q_OTHER : vehicles queued on the opposing group (currently RED)
    ARRIVAL : arrival rate on the GREEN group (veh/min, from virtual detectors)

Each rule: IF Q_THIS is x AND Q_OTHER is y AND ARRIVAL is z THEN GREEN is g
Rules are grouped by the demand on the approach that gets green.
"""
from collections import namedtuple

Rule = namedtuple("Rule", "id group antecedent consequent")

L, M, H = "Low", "Medium", "High"
VS, S, MD, LG, VL = "VeryShort", "Short", "Medium", "Long", "VeryLong"

# (Q_THIS, Q_OTHER, ARRIVAL) -> GREEN
_TABLE = [
    # ---- Group 1: light demand on the green approach (release early) ----
    ("Light demand on green approach", L, L, L, VS),
    ("Light demand on green approach", L, L, M, S),
    ("Light demand on green approach", L, L, H, S),
    ("Light demand on green approach", L, M, L, VS),
    ("Light demand on green approach", L, M, M, VS),
    ("Light demand on green approach", L, M, H, S),
    ("Light demand on green approach", L, H, L, VS),
    ("Light demand on green approach", L, H, M, VS),
    ("Light demand on green approach", L, H, H, VS),
    # ---- Group 2: moderate demand --------------------------------------
    ("Moderate demand on green approach", M, L, L, S),
    ("Moderate demand on green approach", M, L, M, MD),
    ("Moderate demand on green approach", M, L, H, LG),
    ("Moderate demand on green approach", M, M, L, S),
    ("Moderate demand on green approach", M, M, M, MD),
    ("Moderate demand on green approach", M, M, H, MD),
    ("Moderate demand on green approach", M, H, L, VS),
    ("Moderate demand on green approach", M, H, M, S),
    ("Moderate demand on green approach", M, H, H, MD),
    # ---- Group 3: heavy demand (extend green) ---------------------------
    ("Heavy demand on green approach", H, L, L, MD),
    ("Heavy demand on green approach", H, L, M, LG),
    ("Heavy demand on green approach", H, L, H, VL),
    ("Heavy demand on green approach", H, M, L, MD),
    ("Heavy demand on green approach", H, M, M, LG),
    ("Heavy demand on green approach", H, M, H, LG),
    ("Heavy demand on green approach", H, H, L, MD),
    ("Heavy demand on green approach", H, H, M, MD),
    ("Heavy demand on green approach", H, H, H, LG),
]

INPUT_NAMES = ("Q_THIS", "Q_OTHER", "ARRIVAL")

RULES = [
    Rule(i + 1, grp, dict(zip(INPUT_NAMES, (a, b, c))), out)
    for i, (grp, a, b, c, out) in enumerate(_TABLE)
]


def rule_text(rule):
    a = rule.antecedent
    return (f"IF Q_THIS is {a['Q_THIS']} AND Q_OTHER is {a['Q_OTHER']} "
            f"AND ARRIVAL is {a['ARRIVAL']} THEN GREEN is {rule.consequent}")
