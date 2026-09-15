"""Score R-peak detection against the generator's ground truth."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg  # noqa: E402
from src.detection import detect_artifacts  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402
from src.filtering import recover  # noqa: E402
from src.peaks import detect_r_peaks  # noqa: E402
from src.quality import assess_quality  # noqa: E402

TOL = 0.06  # seconds; a detection within +-60 ms counts as a hit


def score(true_idx, det_idx, fs):
    t_true = np.asarray(true_idx) / fs
    t_det = np.asarray(det_idx) / fs
    used = set()
    tp, errs = 0, []
    for t in t_true:
        if t_det.size == 0:
            continue
        d = np.abs(t_det - t)
        j = int(np.argmin(d))
        if d[j] <= TOL and j not in used:
            used.add(j)
            tp += 1
            errs.append((t_det[j] - t) * 1000.0)
    fn = len(t_true) - tp
    fp = len(t_det) - tp
    se = tp / max(1, len(t_true))
    ppv = tp / max(1, len(t_det))
    return tp, fp, fn, se, ppv, (float(np.mean(np.abs(errs))) if errs else float("nan"))


def run(name, rec):
    q = assess_quality(rec.signal, rec.sampling_rate)
    d = detect_artifacts(q, duration=rec.duration)
    r = recover(rec.signal, rec.sampling_rate, d, q)
    pr = detect_r_peaks(r.signal, rec.sampling_rate)
    tp, fp, fn, se, ppv, err = score(rec.r_peaks_true, pr.indices, rec.sampling_rate)
    print(
        f"{name:34s} true={len(rec.r_peaks_true):3d} det={len(pr.beats):3d} "
        f"TP={tp:3d} FP={fp:3d} FN={fn:3d} Se={se:5.3f} PPV={ppv:5.3f} "
        f"|err|={err:5.1f} ms"
    )


def main():
    base = generate_ecg()
    run("clean", base)
    for a in ARTIFACT_TYPES:
        for sev in (0.5, 0.8, 0.95):
            run(
                f"{a} s={sev}",
                corrupt_ecg(base, a, start=8.0, duration=3.0, severity=sev, seed=1),
            )
    for hr in (48, 60, 95, 120):
        run(f"clean HR={hr}", generate_ecg(heart_rate=hr, seed=3))


if __name__ == "__main__":
    main()
