"""Check artifact classification and localisation against injected truth."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg  # noqa: E402
from src.detection import detect_artifacts  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402
from src.quality import assess_quality  # noqa: E402


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def main():
    base = generate_ecg()
    print(f"{'truth':42s} {'detected':30s} {'conf':>5s} {'IoU':>5s}  hit")
    print("-" * 95)
    clean_q = assess_quality(base.signal, base.sampling_rate)
    print(f"{'clean / no artifact':42s} {str(detect_artifacts(clean_q, duration=30.0)):30s}")
    for a in ARTIFACT_TYPES:
        for sev in (0.4, 0.6, 0.8):
            c = corrupt_ecg(base, a, start=8.0, duration=3.0, severity=sev, seed=1)
            q = assess_quality(c.signal, c.sampling_rate)
            dets = detect_artifacts(q, duration=c.duration)
            truth = (8.0, 11.0)
            if not dets:
                print(f"{a + ' s=' + str(sev):42s} {'(none)':30s}")
                continue
            for d in dets:
                ok = "OK" if d.type == a else "MISS"
                print(
                    f"{a + ' s=' + str(sev) + ' @8-11':42s} "
                    f"{d.type + ' ' + str(d.start) + '-' + str(d.end):30s} "
                    f"{d.confidence:5.2f} {iou(truth, (d.start, d.end)):5.2f}  {ok}"
                )


if __name__ == "__main__":
    main()
