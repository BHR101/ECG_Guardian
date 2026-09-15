"""Independent robustness evaluation on real, externally sourced ECG signals.

    python tools/real_data_evaluation.py
    python tools/real_data_evaluation.py --dataset path/to/7dybx7wyfn-3.zip
    python tools/real_data_evaluation.py --limit 100 --json out.json

WHAT THIS IS
------------
The rest of the project is validated against *synthetic* records whose ground
truth we generated ourselves.  That is what makes the per-stage accuracy
numbers in ``src/validation.py`` honest -- and it is also their limitation: the
corruption was injected by the same codebase that is being tested.

This script exists to answer a different, narrower question:

    Does the evidence-gating pipeline behave sensibly on ECG signals that this
    project did not create?

It is deliberately NOT an accuracy benchmark.  See "WHAT THIS CANNOT CLAIM".

THE DATASET
-----------
Mendeley Data 7dybx7wyfn v3 -- "ECG signals (1000 fragments)".
1000 x 10 s single-lead (MLII) fragments drawn from the MIT-BIH Arrhythmia
Database, sorted into 17 rhythm/beat-class folders by the dataset authors.

Each ``.mat`` file holds exactly one variable, ``val``, of shape (1, 3600)
``int16``.  There is no header, no annotation file, no per-sample metadata and
no R-peak marks -- see ``--describe``.

Sampling rate, ADC gain and ADC zero are NOT stored in the files.  They are
taken from the MIT-BIH Arrhythmia Database that the fragments were cut from
(360 Hz, 200 ADU/mV, zero at 1024) and are confirmed empirically: at 360 Hz the
3600 samples are exactly 10.0 s and the normal-sinus fragments land at
physiological rates.  All three are overridable on the command line so the
assumption stays visible rather than baked in.

WHAT THIS CANNOT CLAIM
----------------------
The fragments carry a rhythm/beat class label (the folder name) and nothing
else.  In particular there are no R-peak annotations, no beat timings, no
rhythm onset/offset marks and no signal-quality or artifact annotations.

So this script does NOT and CANNOT report:

  * R-peak sensitivity / PPV / timing error   - no reference peak positions
  * heart-rate or QRS-duration accuracy       - no reference measurements
  * artifact localisation or classification   - no artifact annotations
  * whether a refusal was "correct"           - nothing to compare against

Deriving any of those from the ECG itself would mean scoring the pipeline
against another algorithm's opinion and calling it ground truth.  This script
does not do that.  The class label is used ONLY to group results for reporting;
it is never passed into the pipeline and never used to score a measurement.

WHAT IT DOES REPORT
-------------------
  * execution robustness: crashes, caught stage errors
  * signal-quality distribution on signals we did not generate
  * accept / reject counts and rates per measurement
  * which evidence criteria actually do the rejecting, by name
  * behaviour grouped by the dataset's own rhythm classes
  * determinism: the same fragment twice must give an identical verdict
  * ground-truth isolation, re-proved on real data
  * internal self-consistency of jointly accepted measurements
  * physiological plausibility of the values that were accepted

The dataset is optional.  Nothing else in the project imports this module, and
the Streamlit app never touches it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import warnings
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from src.ecg_generator import ECGRecord  # noqa: E402
from src.evidence import ACCEPTED  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Acquisition constants.  Not stored in the dataset; inherited from MIT-BIH.
# --------------------------------------------------------------------------
MITBIH_FS = 360             # Hz
MITBIH_GAIN = 200.0         # ADU per mV
MITBIH_ADC_ZERO = 1024.0    # ADU at 0 mV

INNER_ZIP_NAME = "ECG signals (1000 fragments).zip"

# Where to look for the dataset when --dataset is not given.
_CANDIDATES = (
    "7dybx7wyfn-3.zip",
    INNER_ZIP_NAME,
    os.path.join("data", "real"),
    os.path.join("data", "real", "7dybx7wyfn-3.zip"),
    os.path.join("data", "real", INNER_ZIP_NAME),
)

DEFAULT_OUT = os.path.join(ROOT, "data", "generated", "real_data_evaluation.json")

# Measurements reported, in display order.
MEASUREMENTS = ("heart_rate", "rr_interval", "rr_consistency", "qrs_duration")

# Physiological rails used ONLY to sanity-check accepted values.  These are not
# gates and nothing is scored against them; a value outside them would be a
# finding about this pipeline, not about the patient.
PLAUSIBLE_HR_BPM = (20.0, 250.0)
PLAUSIBLE_QRS_MS = (40.0, 200.0)


# --------------------------------------------------------------------------
# Dataset access
# --------------------------------------------------------------------------
class Fragment:
    """One 10 s fragment: raw samples plus the dataset's own labelling."""

    __slots__ = ("key", "cls", "source_record", "raw")

    def __init__(self, key: str, cls: str, source_record: str, raw: np.ndarray):
        self.key = key
        self.cls = cls
        self.source_record = source_record
        self.raw = raw


def find_dataset(explicit: str | None) -> str | None:
    """Locate the dataset, or return None if it is not present."""
    if explicit:
        return explicit if os.path.exists(explicit) else None
    env = os.environ.get("ECG_GUARDIAN_REAL_DATA")
    if env and os.path.exists(env):
        return env
    for rel in _CANDIDATES:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    return None


def _parse_member(name: str) -> tuple[str, str] | None:
    """Map ``MLII/7 PVC/106m (3).mat`` -> ("PVC", "106m"), or None."""
    if not name.lower().endswith(".mat"):
        return None
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return None
    folder, leaf = parts[-2], parts[-1]
    # Folder names are "<ordinal> <CLASS>"; keep the class only.
    cls = re.sub(r"^\d+\s+", "", folder).strip() or folder
    record = re.sub(r"\s*\(\d+\)\.mat$", "", leaf).strip() or leaf
    return cls, record


def _decode_mat(blob: bytes, key: str) -> np.ndarray | None:
    """Pull the single signal variable out of one .mat file."""
    try:
        mat = loadmat(io.BytesIO(blob))
    except Exception:
        return None
    arrays = [
        np.asarray(v, dtype=float).ravel()
        for k, v in mat.items()
        if not k.startswith("__") and hasattr(v, "shape") and np.asarray(v).size
    ]
    if not arrays:
        return None
    # The published files carry exactly one variable, ``val``.  If a future
    # revision carries more, take the longest rather than guessing a name.
    return max(arrays, key=lambda a: a.size)


def load_fragments(path: str, limit: int | None = None) -> list[Fragment]:
    """Load fragments from a nested zip, a flat zip, or an extracted folder."""
    items: list[tuple[str, bytes]] = []

    if os.path.isdir(path):
        for dirpath, _dirs, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".mat"):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, path).replace("\\", "/")
                    with open(full, "rb") as fh:
                        items.append((rel, fh.read()))
    else:
        with zipfile.ZipFile(path) as outer:
            names = outer.namelist()
            inner_names = [n for n in names if n.lower().endswith(".zip")]
            if inner_names and not any(n.lower().endswith(".mat") for n in names):
                # The Mendeley download wraps the data in a second zip.
                with zipfile.ZipFile(io.BytesIO(outer.read(inner_names[0]))) as inner:
                    for n in inner.namelist():
                        if n.lower().endswith(".mat"):
                            items.append((n, inner.read(n)))
            else:
                for n in names:
                    if n.lower().endswith(".mat"):
                        items.append((n, outer.read(n)))

    items.sort(key=lambda kv: kv[0])
    fragments: list[Fragment] = []
    for name, blob in items:
        parsed = _parse_member(name)
        if parsed is None:
            continue
        cls, source_record = parsed
        raw = _decode_mat(blob, name)
        if raw is None or raw.size < 2:
            continue
        fragments.append(Fragment(name, cls, source_record, raw))
        if limit is not None and len(fragments) >= limit:
            break
    return fragments


def to_record(frag: Fragment, fs: int, gain: float, adc_zero: float) -> ECGRecord:
    """Wrap raw ADC counts as an ECGRecord in millivolts.

    ``r_peaks_true`` is empty because this dataset has no R-peak annotations.
    The pipeline never reads it -- ``--check-isolation`` re-proves that here.
    """
    mv = (np.asarray(frag.raw, dtype=float) - adc_zero) / gain
    return ECGRecord(
        time=np.arange(mv.size, dtype=float) / fs,
        signal=mv,
        sampling_rate=fs,
        r_peaks_true=np.asarray([], dtype=int),
        metadata={"source": "mendeley-7dybx7wyfn-v3", "fragment": frag.key},
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def describe(fragments: list[Fragment], path: str, fs: int) -> dict:
    """Report what the dataset actually contains, without running anything."""
    lengths = Counter(int(f.raw.size) for f in fragments)
    by_class = Counter(f.cls for f in fragments)
    records_by_class = defaultdict(set)
    for f in fragments:
        records_by_class[f.cls].add(f.source_record)
    all_records = {f.source_record for f in fragments}

    print("=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)
    print(f"  source              : {path}")
    print(f"  fragments           : {len(fragments)}")
    print(f"  distinct source recs: {len(all_records)}")
    print(f"  sample counts       : "
          f"{', '.join(f'{n} x{c}' for n, c in lengths.most_common(5))}")
    print(f"  assumed rate        : {fs} Hz "
          f"-> {list(lengths)[0] / fs:.2f} s per fragment")
    print(f"  variables per file  : 1 (int16 signal, no header, no annotations)")
    print()
    print(f"  {'class':<12}{'fragments':>11}{'source records':>16}")
    print(f"  {'-' * 12}{'-' * 11:>11}{'-' * 16:>16}")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<12}{n:>11}{len(records_by_class[cls]):>16}")
    print()
    print("  ANNOTATIONS PRESENT")
    print("    rhythm / beat class per fragment : YES (folder name)")
    print("    R-peak positions                 : NO")
    print("    beat-level annotations           : NO")
    print("    artifact / noise annotations      : NO")
    print("    signal-quality annotations        : NO")
    print("    reference HR / QRS measurements   : NO")
    return {
        "path": path,
        "n_fragments": len(fragments),
        "n_source_records": len(all_records),
        "samples_per_fragment": dict(lengths),
        "assumed_sampling_rate_hz": fs,
        "fragments_per_class": dict(by_class),
        "source_records_per_class": {k: len(v) for k, v in records_by_class.items()},
        "annotations": {
            "rhythm_beat_class_per_fragment": True,
            "r_peak_positions": False,
            "beat_annotations": False,
            "artifact_annotations": False,
            "quality_annotations": False,
            "reference_measurements": False,
        },
    }


def failed_criteria(measurement) -> list[str]:
    """Names of the evidence criteria this measurement did not meet."""
    return [c.name for c in measurement.criteria if not c.passed]


def evaluate(fragments, fs, gain, adc_zero, verbose=False) -> dict:
    """Run the pipeline over every fragment and collect behaviour, not scores."""
    rows: list[dict] = []
    crashes: list[dict] = []

    for i, frag in enumerate(fragments):
        record = to_record(frag, fs, gain, adc_zero)
        try:
            # The class label is NOT passed in; the scenario string is the
            # fragment id only, so nothing about the label can reach a gate.
            result = run_pipeline(record, frag.key)
        except Exception as exc:  # a crash is itself the finding
            crashes.append({"fragment": frag.key, "error": f"{type(exc).__name__}: {exc}"})
            rows.append({"fragment": frag.key, "cls": frag.cls, "crashed": True})
            continue

        row = {
            "fragment": frag.key,
            "cls": frag.cls,
            "source_record": frag.source_record,
            "crashed": False,
            "stage_errors": list(result.errors),
            "quality_raw": round(result.quality_raw.score, 2),
            "quality_processed": round(result.quality_processed.score, 2),
            "recovery_status": result.recovery.status,
            "n_detections": len(result.detections),
            "detected_classes": [d.label for d in result.detections],
            "n_candidate_beats": len(result.beats),
            "overall_trust": result.overall_trust,
        }
        for name in MEASUREMENTS:
            m = result.measurement(name)
            if m is None:
                continue
            row[f"{name}_status"] = m.status
            row[f"{name}_value"] = m.value
            row[f"{name}_confidence"] = round(m.confidence, 3)
            row[f"{name}_validated_beats"] = m.validated_beats
            row[f"{name}_failed_criteria"] = failed_criteria(m)
        rows.append(row)

        if verbose and (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{len(fragments)} fragments", file=sys.stderr)

    return {"rows": rows, "crashes": crashes}


def determinism_check(fragments, fs, gain, adc_zero, n: int = 12) -> dict:
    """The same fragment analysed twice must produce an identical verdict."""
    picks = fragments[:: max(1, len(fragments) // max(1, n))][:n]
    mismatches = []
    for frag in picks:
        a = run_pipeline(to_record(frag, fs, gain, adc_zero), frag.key)
        b = run_pipeline(to_record(frag, fs, gain, adc_zero), frag.key)
        sa = [(m.measurement, m.status, m.value, m.confidence) for m in a.measurements]
        sb = [(m.measurement, m.status, m.value, m.confidence) for m in b.measurements]
        if sa != sb or round(a.quality_raw.score, 9) != round(b.quality_raw.score, 9):
            mismatches.append(frag.key)
    return {"n_checked": len(picks), "mismatches": mismatches,
            "deterministic": not mismatches}


def isolation_check(fragments, fs, gain, adc_zero, n: int = 8) -> dict:
    """Re-prove ground-truth isolation on real signals.

    This dataset has no ground truth, so ``r_peaks_true`` is empty.  Filling it
    with deliberately wrong values must change nothing at all.
    """
    picks = fragments[:: max(1, len(fragments) // max(1, n))][:n]
    differences = []
    for frag in picks:
        clean = to_record(frag, fs, gain, adc_zero)
        a = run_pipeline(clean, frag.key)
        poisoned = ECGRecord(
            time=clean.time,
            signal=clean.signal,
            sampling_rate=clean.sampling_rate,
            # Deliberate nonsense: evenly spaced "peaks" at 1 Hz.
            r_peaks_true=np.arange(0, clean.signal.size, fs, dtype=int),
            metadata=dict(clean.metadata),
        )
        b = run_pipeline(poisoned, frag.key)
        sa = [(m.measurement, m.status, m.value, m.confidence) for m in a.measurements]
        sb = [(m.measurement, m.status, m.value, m.confidence) for m in b.measurements]
        if sa != sb:
            differences.append(frag.key)
    return {"n_checked": len(picks), "differences": differences,
            "isolated": not differences}


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def summarise(data: dict, dataset_info: dict, det: dict, iso: dict) -> dict:
    """Turn per-fragment rows into the reported summary."""
    rows = data["rows"]
    ok = [r for r in rows if not r["crashed"]]
    n_total, n_ok = len(rows), len(ok)

    quality = np.asarray([r["quality_raw"] for r in ok], dtype=float)
    stage_err = [r for r in ok if r["stage_errors"]]

    per_measurement: dict[str, dict] = {}
    for name in MEASUREMENTS:
        statuses = [r.get(f"{name}_status") for r in ok if f"{name}_status" in r]
        acc = sum(1 for s in statuses if s == ACCEPTED)
        rej = len(statuses) - acc
        reasons = Counter()
        for r in ok:
            if r.get(f"{name}_status") and r[f"{name}_status"] != ACCEPTED:
                reasons.update(r.get(f"{name}_failed_criteria") or ["(confidence only)"])
        values = [r[f"{name}_value"] for r in ok
                  if r.get(f"{name}_status") == ACCEPTED
                  and r.get(f"{name}_value") is not None]
        per_measurement[name] = {
            "n": len(statuses),
            "accepted": acc,
            "rejected": rej,
            "acceptance_rate_pct": round(_pct(acc, len(statuses)), 2),
            "top_failed_criteria": reasons.most_common(6),
            "accepted_values": {
                "n": len(values),
                "min": round(float(np.min(values)), 2) if values else None,
                "median": round(float(np.median(values)), 2) if values else None,
                "max": round(float(np.max(values)), 2) if values else None,
            },
        }

    # Plausibility of what was accepted (a check on us, not on the patient).
    implausible = []
    for r in ok:
        hr = r.get("heart_rate_value")
        if r.get("heart_rate_status") == ACCEPTED and hr is not None:
            if not (PLAUSIBLE_HR_BPM[0] <= hr <= PLAUSIBLE_HR_BPM[1]):
                implausible.append({"fragment": r["fragment"], "measurement": "heart_rate",
                                    "value": hr})
        q = r.get("qrs_duration_value")
        if r.get("qrs_duration_status") == ACCEPTED and q is not None:
            if not (PLAUSIBLE_QRS_MS[0] <= q <= PLAUSIBLE_QRS_MS[1]):
                implausible.append({"fragment": r["fragment"], "measurement": "qrs_duration",
                                    "value": q})

    # Self-consistency: where HR and RR interval were BOTH accepted, they are
    # derived from the same intervals and must agree.  This is an internal
    # coherence check -- it says nothing about whether either is correct.
    deltas, inconsistent = [], []
    for r in ok:
        if (r.get("heart_rate_status") == ACCEPTED
                and r.get("rr_interval_status") == ACCEPTED
                and r.get("heart_rate_value") and r.get("rr_interval_value")):
            implied = 60000.0 / r["rr_interval_value"]   # RR is reported in ms
            delta = abs(implied - r["heart_rate_value"])
            deltas.append(delta)
            if delta > 1.0:
                inconsistent.append({"fragment": r["fragment"], "delta_bpm": round(delta, 3)})

    # Behaviour grouped by the dataset's own classes -- descriptive only.
    by_class: dict[str, dict] = {}
    classes = sorted({r["cls"] for r in ok})
    for cls in classes:
        sub = [r for r in ok if r["cls"] == cls]
        entry = {"n": len(sub),
                 "median_quality": round(float(np.median(
                     [r["quality_raw"] for r in sub])), 1)}
        for name in MEASUREMENTS:
            st = [r.get(f"{name}_status") for r in sub if f"{name}_status" in r]
            entry[f"{name}_acceptance_rate_pct"] = round(
                _pct(sum(1 for s in st if s == ACCEPTED), len(st)), 1)
        by_class[cls] = entry

    detected = Counter()
    for r in ok:
        detected.update(r["detected_classes"] or ["(none)"])

    return {
        "dataset": dataset_info,
        "execution": {
            "fragments_processed": n_total,
            "fragments_succeeded": n_ok,
            "pipeline_crashes": len(data["crashes"]),
            "crash_detail": data["crashes"][:10],
            "fragments_with_caught_stage_errors": len(stage_err),
        },
        "quality": {
            "median": round(float(np.median(quality)), 1) if n_ok else None,
            "p05": round(float(np.percentile(quality, 5)), 1) if n_ok else None,
            "p95": round(float(np.percentile(quality, 95)), 1) if n_ok else None,
            "min": round(float(np.min(quality)), 1) if n_ok else None,
            "max": round(float(np.max(quality)), 1) if n_ok else None,
        },
        "measurements": per_measurement,
        "artifact_classes_detected": detected.most_common(),
        "by_rhythm_class": by_class,
        "determinism": det,
        "ground_truth_isolation": iso,
        "accepted_value_plausibility": {
            "n_implausible": len(implausible),
            "detail": implausible[:10],
            "hr_rails_bpm": list(PLAUSIBLE_HR_BPM),
            "qrs_rails_ms": list(PLAUSIBLE_QRS_MS),
        },
        "hr_rr_self_consistency": {
            "n_jointly_accepted": len(deltas),
            "max_delta_bpm": round(float(np.max(deltas)), 4) if deltas else None,
            "n_inconsistent_over_1bpm": len(inconsistent),
            "detail": inconsistent[:10],
        },
        "not_evaluated": {
            "r_peak_sensitivity_ppv_timing": "no R-peak annotations in this dataset",
            "heart_rate_accuracy": "no reference heart rate in this dataset",
            "qrs_duration_accuracy": "no reference QRS duration in this dataset",
            "artifact_localisation_and_class": "no artifact annotations in this dataset",
            "refusal_correctness": "no reference measurements to compare a refusal against",
        },
    }


def print_report(s: dict) -> None:
    ex, q = s["execution"], s["quality"]
    print()
    print("=" * 70)
    print("REAL DATASET EVALUATION")
    print("=" * 70)
    print(f"  dataset            : {s['dataset']['path']}")
    print(f"  fragments          : {s['dataset']['n_fragments']} "
          f"from {s['dataset']['n_source_records']} source recordings")
    print(f"  sampling rate used : {s['dataset']['assumed_sampling_rate_hz']} Hz")
    print()
    print(f"Records processed            : {ex['fragments_processed']}")
    print(f"Records successfully processed: {ex['fragments_succeeded']}")
    print(f"Pipeline crashes             : {ex['pipeline_crashes']}")
    print(f"Caught stage errors          : {ex['fragments_with_caught_stage_errors']}")

    print()
    print("Signal quality (raw, prototype index 0-100):")
    print(f"  median {q['median']}   p05 {q['p05']}   p95 {q['p95']}   "
          f"range {q['min']} - {q['max']}")

    for name in MEASUREMENTS:
        m = s["measurements"].get(name)
        if not m:
            continue
        print()
        print(f"{name.replace('_', ' ').upper()}:")
        print(f"  accepted        : {m['accepted']}")
        print(f"  rejected        : {m['rejected']}")
        print(f"  acceptance rate : {m['acceptance_rate_pct']:.1f}%")
        av = m["accepted_values"]
        if av["n"]:
            print(f"  accepted values : median {av['median']} "
                  f"(range {av['min']} - {av['max']})")
        if m["top_failed_criteria"]:
            print("  criteria that did the rejecting:")
            for crit, n in m["top_failed_criteria"]:
                print(f"      {n:5d}  {crit}")

    print()
    print("Artifact classes localised across the corpus:")
    for label, n in s["artifact_classes_detected"]:
        print(f"    {n:5d}  {label}")

    print()
    print("Behaviour by the dataset's own rhythm class (DESCRIPTIVE ONLY --")
    print("these labels are not ground truth for any measurement):")
    print(f"    {'class':<12}{'n':>5}{'median Q':>10}{'HR acc%':>9}{'QRS acc%':>10}")
    for cls, e in sorted(s["by_rhythm_class"].items(),
                         key=lambda kv: -kv[1]["heart_rate_acceptance_rate_pct"]):
        print(f"    {cls:<12}{e['n']:>5}{e['median_quality']:>10.1f}"
              f"{e['heart_rate_acceptance_rate_pct']:>9.1f}"
              f"{e['qrs_duration_acceptance_rate_pct']:>10.1f}")

    print()
    print("Integrity checks:")
    d, i = s["determinism"], s["ground_truth_isolation"]
    print(f"  determinism           : {'PASS' if d['deterministic'] else 'FAIL'} "
          f"({d['n_checked']} fragments run twice)")
    print(f"  ground-truth isolation: {'PASS' if i['isolated'] else 'FAIL'} "
          f"({i['n_checked']} fragments re-run with deliberately wrong ground truth)")
    p = s["accepted_value_plausibility"]
    print(f"  accepted values within physiological rails: "
          f"{'PASS' if not p['n_implausible'] else 'FAIL'} "
          f"({p['n_implausible']} outside)")
    c = s["hr_rr_self_consistency"]
    if c["n_jointly_accepted"]:
        print(f"  HR / RR self-consistency  : {c['n_jointly_accepted']} jointly "
              f"accepted, max disagreement {c['max_delta_bpm']} BPM")
    else:
        print("  HR / RR self-consistency  : no fragment had both accepted")

    print()
    print("NOT EVALUATED (ground truth unavailable in this dataset):")
    for k, why in s["not_evaluated"].items():
        print(f"    {k:<34} - {why}")
    print()
    print("This is a robustness and behaviour evaluation, not an accuracy")
    print("benchmark and not a clinical validation. The dataset provides a")
    print("rhythm class per fragment and nothing else; no measurement reported")
    print("here has been scored against a reference value.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Independent real-ECG robustness evaluation (optional dataset).")
    ap.add_argument("--dataset", default=None,
                    help="zip or folder holding the fragments")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N fragments")
    ap.add_argument("--fs", type=int, default=MITBIH_FS)
    ap.add_argument("--gain", type=float, default=MITBIH_GAIN)
    ap.add_argument("--adc-zero", type=float, default=MITBIH_ADC_ZERO)
    ap.add_argument("--json", default=DEFAULT_OUT, help="where to write the JSON summary")
    ap.add_argument("--describe", action="store_true",
                    help="report dataset structure and exit without running")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = find_dataset(args.dataset)
    if path is None:
        print("Real-ECG dataset not found -- skipping.")
        print()
        print("  This evaluation is optional. Nothing else in the project needs")
        print("  it: the tests, the validation suite, the adversarial corpus and")
        print("  the Streamlit app all run without it.")
        print()
        print("  To run it, download Mendeley Data 7dybx7wyfn v3")
        print("  (https://data.mendeley.com/datasets/7dybx7wyfn/3) and either")
        print("  place 7dybx7wyfn-3.zip in the project root, or point at it:")
        print()
        print("      python tools/real_data_evaluation.py --dataset <path>")
        return 0

    fragments = load_fragments(path, limit=args.limit)
    if not fragments:
        print(f"No .mat fragments found in {path}")
        return 1

    info = describe(fragments, path, args.fs)
    if args.describe:
        return 0

    if not args.quiet:
        print()
        print(f"Running the pipeline over {len(fragments)} fragments...")
    data = evaluate(fragments, args.fs, args.gain, args.adc_zero,
                    verbose=not args.quiet)
    det = determinism_check(fragments, args.fs, args.gain, args.adc_zero)
    iso = isolation_check(fragments, args.fs, args.gain, args.adc_zero)
    summary = summarise(data, info, det, iso)

    print_report(summary)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_fragment": data["rows"]}, fh, indent=2)
    print(f"\nwritten to {args.json}")

    # Only a crash, a determinism failure or a ground-truth leak is a failure
    # here.  A refusal is not an error -- refusing is what this system does
    # when the evidence is not there.
    bad = (summary["execution"]["pipeline_crashes"]
           or not det["deterministic"]
           or not iso["isolated"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
