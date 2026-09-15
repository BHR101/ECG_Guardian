"""Write one example file per supported upload format.

    python tools/make_sample_uploads.py

Files land in ``data/samples/``.  They exist so the upload feature can be
demonstrated without hunting for a recording, and so each reader can be
exercised against a file rather than only against an in-memory buffer.

The signals are the project's own synthetic scenarios, so what each file should
produce is already known -- which is what makes them useful as a check.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from src.artifacts import corrupt_ecg  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "samples"
)


def write_csv_with_time(path, rec):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("time,ecg_mV\n")
        for t, v in zip(rec.time, rec.signal):
            fh.write(f"{t:.6f},{v:.6f}\n")


def write_csv_bare(path, rec):
    """No header, no time column -- the app must ask for the rate."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for v in rec.signal:
            fh.write(f"{v:.6f}\n")


def write_csv_multilead(path, rec):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("time,Lead I,Lead II,Lead III\n")
        for t, v in zip(rec.time, rec.signal):
            fh.write(f"{t:.6f},{v:.6f},{0.7 * v:.6f},{-v:.6f}\n")


def write_edf(path, rec, label="ECG II"):
    """A minimal, standards-conformant EDF."""
    fs = rec.sampling_rate
    spr, record_sec = fs, 1.0
    n_rec = len(rec.signal) // spr
    sig = rec.signal[: n_rec * spr]
    pmin, pmax, dmin, dmax = -5.0, 5.0, -32768, 32767

    header = (
        f"{0:<8}{'X X X Sample':<80}{'Startdate 01-JAN-2024 X X ECG Guardian':<80}"
        f"{'01.01.24':<8}{'00.00.00':<8}{512:<8}{'EDF+C':<44}"
        f"{n_rec:<8}{record_sec:<8}{1:<4}"
    ).encode("ascii")
    assert len(header) == 256

    for value, width in (
        (label, 16), ("", 80), ("mV", 8), (f"{pmin:g}", 8), (f"{pmax:g}", 8),
        (f"{dmin:d}", 8), (f"{dmax:d}", 8), ("", 80), (f"{spr:d}", 8), ("", 32),
    ):
        header += f"{value:<{width}}".encode("ascii")

    dig = np.round((sig - pmin) * (dmax - dmin) / (pmax - pmin) + dmin)
    body = np.clip(dig, dmin, dmax).astype("<i2").tobytes()
    with open(path, "wb") as fh:
        fh.write(header + body)


def write_wfdb(stem, rec, gain=200.0, baseline=0):
    """WFDB format 16, the straightforward PhysioNet layout."""
    name = os.path.basename(stem)
    adc = np.clip(np.round(rec.signal * gain + baseline), -32768, 32767).astype("<i2")
    with open(stem + ".dat", "wb") as fh:
        fh.write(adc.tobytes())
    with open(stem + ".hea", "w", encoding="utf-8") as fh:
        fh.write(f"{name} 1 {rec.sampling_rate} {adc.size}\n")
        fh.write(f"{name}.dat 16 {gain:g}({baseline:d})/mV 16 {baseline:d} "
                 f"{int(adc[0])} 0 MLII\n")


def write_mat(path, rec):
    from scipy.io import savemat
    savemat(path, {"val": rec.signal.reshape(1, -1),
                   "fs": float(rec.sampling_rate)})


def write_npy(path, rec):
    np.save(path, rec.signal)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    clean = generate_ecg(seed=42)
    noisy = corrupt_ecg(generate_ecg(seed=42), "muscle_noise", severity=0.6, seed=13)

    write_csv_with_time(os.path.join(OUT, "clean_with_time.csv"), clean)
    write_csv_bare(os.path.join(OUT, "clean_no_rate.csv"), clean)
    write_csv_multilead(os.path.join(OUT, "three_lead.csv"), clean)
    write_edf(os.path.join(OUT, "clean.edf"), clean)
    write_wfdb(os.path.join(OUT, "sample100"), clean)
    write_mat(os.path.join(OUT, "clean.mat"), clean)
    write_npy(os.path.join(OUT, "clean.npy"), clean)
    write_csv_with_time(os.path.join(OUT, "muscle_noise_with_time.csv"), noisy)

    print(f"Sample upload files written to {OUT}\n")
    for fn in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, fn))
        print(f"  {fn:<28}{size / 1024:8.1f} KB")
    print()
    print("What to expect when you upload them:")
    print("  clean_with_time.csv        rate read from the time column; all accepted")
    print("  clean_no_rate.csv          no rate in the file -- the app asks for it")
    print("  three_lead.csv             three leads offered; Lead III is inverted")
    print("  clean.edf / .mat           rate read from the file; all accepted")
    print("  sample100.hea + .dat       upload BOTH; rate and gain read from header")
    print("  clean.npy                  no rate in the file -- the app asks for it")
    print("  muscle_noise_with_time.csv heart rate ACCEPTED, QRS NOT REPORTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
