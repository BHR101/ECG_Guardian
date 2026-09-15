"""Check that recovery improves what it should and fails where it should."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg  # noqa: E402
from src.detection import detect_artifacts  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402
from src.filtering import recover  # noqa: E402
from src.quality import assess_quality  # noqa: E402


def main():
    base = generate_ecg()
    print(f"{'case':36s} {'glob':>12s} {'region':>14s}  {'status':10s} method")
    print("-" * 110)
    for a in ARTIFACT_TYPES:
        for sev in (0.4, 0.6, 0.8, 0.95):
            c = corrupt_ecg(base, a, start=8.0, duration=3.0, severity=sev, seed=1)
            q = assess_quality(c.signal, c.sampling_rate)
            dets = detect_artifacts(q, duration=c.duration)
            r = recover(c.signal, c.sampling_rate, dets, q)
            gl = f"{r.global_before:5.1f}->{r.global_after:5.1f}"
            rg = (
                f"{r.affected_before:5.1f}->{r.affected_after:5.1f}"
                if r.regions
                else "      n/a     "
            )
            print(
                f"{a + ' s=' + str(sev):36s} {gl:>12s} {rg:>14s}  "
                f"{r.status:10s} {r.method_label[:46]}"
            )
    q = assess_quality(base.signal, base.sampling_rate)
    r = recover(base.signal, base.sampling_rate, detect_artifacts(q, duration=30.0), q)
    print(f"{'clean':36s} {r.global_before:5.1f}->{r.global_after:5.1f}        "
          f"        {r.status}")


if __name__ == "__main__":
    main()
