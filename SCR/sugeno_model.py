"""
sugeno_model.py
===============
Zero-order Takagi-Sugeno FIS used as a benchmark against the Mamdani FIS.

It re-uses the SAME fuzzification and the SAME 27 rules; the only difference
is the consequent: each output term is a crisp constant (the peak of the
Mamdani output set) and the result is a weighted average of rule strengths:

        green = sum(w_i * z_i) / sum(w_i)
"""
from .fuzzy_system import repair, DEFAULT_PARAMS, build_variables
from .fuzzification import fuzzify_inputs
from .fuzzy_rules import RULES, INPUT_NAMES


class SugenoFIS:
    def __init__(self, params=None, rules=None):
        self.params = repair(DEFAULT_PARAMS if params is None else params)
        self.inputs, self.output = build_variables(self.params)
        self.rules = list(RULES if rules is None else rules)
        c = self.params[9:14]
        self.constants = {"VeryShort": c[0], "Short": c[1], "Medium": c[2],
                          "Long": c[3], "VeryLong": c[4]}

    def infer(self, q_this, q_other, arrival, details=False):
        crisp = {"Q_THIS": q_this, "Q_OTHER": q_other, "ARRIVAL": arrival}
        fz = fuzzify_inputs(self.inputs, crisp)
        num = den = 0.0
        strengths = []
        for r in self.rules:
            w = min(fz[v][r.antecedent[v]] for v in INPUT_NAMES)
            strengths.append(w)
            num += w * self.constants[r.consequent]
            den += w
        green = num / den if den > 1e-12 else 0.5 * sum(self.output.universe)
        if not details:
            return green
        return {"green": green, "fuzzified": fz, "strengths": strengths}

    def infer_from_state(self, state):
        return self.infer(state["q_this"], state["q_other"], state["arrival_this"])
