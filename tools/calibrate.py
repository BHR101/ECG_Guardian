"""Print measured quality features for clean and corrupted signals.

Used to set the penalty knees in src/quality.py from real data rather than
guesswork.  Not part of the runtime pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402
from src.quality import assess_quality  # noqa: E402

KEYS = [
    "baseline_drift_ratio",
    "hf_noise_ratio",
    "powerline_ratio",
    "amplitude_ratio",
    "dropout_index",
    "qrs_visibility",
]


def show(name, rec, t0=None, t1=None):
    q = assess_quality(rec.signal, rec.sampling_rate)
    if t0 is None:
        ws = q.windows
    else:
        ws = [w for w in q.windows if w.end > t0 and w.start < t1]
    vals = [np.mean([w.features[k] for w in ws]) for k in KEYS]
    sc = np.mean([w.score for w in ws])
    print(f"{name:34s} " + " ".join(f"{v:11.4f}" for v in vals) + f" {sc:7.1f}")


def main():
    base = generate_ecg()
    header = f"{'case':34s} " + " ".join(f"{k[:11]:>11s}" for k in KEYS) + f" {'score':>7s}"
    print(header)
    print("-" * len(header))
    show("clean (whole record)", base)
    for a in ARTIFACT_TYPES:
        for sev in (0.35, 0.8):
            c = corrupt_ecg(base, a, start=8.0, duration=3.0, severity=sev, seed=1)
            show(f"{a[:17]} s={sev} @8-11s", c, 8.0, 11.0)
            show(f"{a[:17]} s={sev} @outside", c, 15.0, 25.0)


if __name__ == "__main__":
    main()
