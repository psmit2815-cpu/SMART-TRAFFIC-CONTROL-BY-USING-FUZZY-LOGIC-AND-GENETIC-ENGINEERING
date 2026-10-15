"""
generate_dataset.py
===================
Creates data/traffic_data.csv : 3,000 synthetic intersection states.

Columns
    q_this        vehicles queued on the approach group about to get green   (0-60)
    q_other       vehicles queued on the opposing group                      (0-60)
    arrival_this  arrival rate on the green group (veh/min)                  (0-40)
    reference_green  queue-clearance reference (s)

reference_green is a transparent CRISP heuristic, NOT field ground truth:
    time to discharge the queue (group saturation flow 1.0 veh/s, minus the
    arrival rate) + 2 s start-up loss, shortened when the opposing queue is
    large relative to this one, clipped to [10, 60] s.
It is used only to check that the FIS output follows a sensible trend.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "traffic_data.csv")


def reference_green(q_this, q_other, arrival):
    free = np.maximum(1.0 - arrival / 60.0, 0.15)
    clear = 2.0 + q_this / free
    damp = 1.0 - 0.35 * q_other / (q_other + q_this + 1.0)
    return np.clip(clear * damp, 10.0, 60.0)


def generate(n=3000, seed=7, path=OUT):
    rng = np.random.default_rng(seed)
    q_this = np.round(rng.beta(1.6, 2.4, n) * 60, 1)
    q_other = np.round(rng.beta(1.6, 2.4, n) * 60, 1)
    arrival = np.round(np.clip(0.2 * q_this + rng.normal(8, 6, n), 0, 40), 1)
    df = pd.DataFrame({"q_this": q_this, "q_other": q_other, "arrival_this": arrival})
    df["reference_green"] = np.round(reference_green(q_this, q_other, arrival), 2)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = generate()
    print(f"Saved {len(d)} records -> {OUT}")
    print(d.describe().round(2))
