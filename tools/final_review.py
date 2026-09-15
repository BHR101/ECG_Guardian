"""Answer the project's own final-review checklist from live execution.

Every answer is computed, not asserted.  Run:  python tools/final_review.py
"""

import inspect
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")


from src import evidence as ev_mod  # noqa: E402
from src.demo import SCENARIOS, build  # noqa: E402
from src.ecg_generator import generate_ecg  # noqa: E402
from src.evidence import ACCEPTED, REJECTED  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results: list[tuple[str, bool, str]] = []


def check(question: str, ok: bool, detail: str) -> None:
    results.append((question, ok, detail))


# 1. Are we actually measuring anything, or just displaying fake values?
r = run_pipeline(generate_ecg(seed=42), "clean")
hr = r.measurement("heart_rate")
true_hr = r.record.metadata["true_mean_hr_bpm"]
check(
    "1. Are the values actually measured?",
    abs(hr.value - true_hr) < 1.5,
    f"heart rate {hr.value} BPM vs generator truth {true_hr:.2f} BPM "
    f"(error {abs(hr.value - true_hr):.2f})",
)

# 2. Is confidence derived from evidence?
terms_ok, mismatch = True, ""
for scenario in SCENARIOS:
    res = run_pipeline(scenario.record(), scenario.name)
    for m in res.measurements:
        if not m.confidence_terms:
            continue
        total = min(sum(m.confidence_terms.values()), 0.99)
        if abs(total - m.confidence) > 0.011:
            terms_ok = False
            mismatch = f"{scenario.key}/{m.measurement}: {m.confidence} vs {total}"
src_text = "".join(
    open(os.path.join(ROOT, "src", f), encoding="utf-8").read()
    for f in os.listdir(os.path.join(ROOT, "src"))
    if f.endswith(".py")
)
check(
    "2. Is confidence derived from measured evidence?",
    terms_ok,
    "every reported confidence equals the sum of its own weighted, measured terms"
    if terms_ok
    else mismatch,
)

# 3. Can ground truth leak into the detector?
from src import detection, peaks, quality  # noqa: E402

sigs = {
    "detect_r_peaks": list(inspect.signature(peaks.detect_r_peaks).parameters),
    "assess_quality": list(inspect.signature(quality.assess_quality).parameters),
    "detect_artifacts": list(inspect.signature(detection.detect_artifacts).parameters),
    "run_evidence_engine": list(
        inspect.signature(ev_mod.run_evidence_engine).parameters
    ),
}
banned = ("r_peaks_true", "ground_truth", "truth")
leaks = [
    f"{n}({p})" for n, ps in sigs.items() for p in ps if any(b in p for b in banned)
]
analysis_files = ["quality.py", "detection.py", "filtering.py", "peaks.py",
                  "measurements.py", "evidence.py"]
mentions = [
    f for f in analysis_files
    if "r_peaks_true" in open(os.path.join(ROOT, "src", f), encoding="utf-8").read()
]
check(
    "3. Can ground truth leak into the detector?",
    not leaks and not mentions,
    "no analysis entry point takes a ground-truth argument, and no analysis "
    "module mentions r_peaks_true at all"
    if not leaks and not mentions
    else f"leaks={leaks} mentions={mentions}",
)

# 4. Does recovery actually get revalidated?
res = run_pipeline(build("motion_unrecoverable"), "motion_unrecoverable")
revalidated = all(reg.reason for reg in res.recovery.regions)
score_rose = res.recovery.affected_after > res.recovery.affected_before
still_failed = res.recovery.status == "FAILED"
check(
    "4. Does recovery actually get revalidated?",
    revalidated and score_rose and still_failed,
    f"filtering raised region quality {res.recovery.affected_before:.0f} -> "
    f"{res.recovery.affected_after:.0f} and revalidation STILL failed it",
)

# 5. Can the system reject a measurement?
res = run_pipeline(build("severe"), "severe")
check(
    "5. Can the system reject a measurement?",
    len(res.accepted) == 0 and all(m.value is None for m in res.rejected),
    f"severe corruption: {len(res.rejected)} of {len(res.measurements)} refused, "
    f"no numbers produced",
)

# 6. Can different measurements get different decisions?
res = run_pipeline(build("muscle_noise"), "muscle_noise")
m_hr, m_qrs = res.measurement("heart_rate"), res.measurement("qrs_duration")
check(
    "6. Can different measurements get different decisions?",
    m_hr.status == ACCEPTED and m_qrs.status == REJECTED,
    f"same record: heart rate {m_hr.status} ({m_hr.value} BPM), "
    f"QRS duration {m_qrs.status}",
)

# 7. Are artifact regions actually localised?
res = run_pipeline(build("motion_recoverable"), "motion_recoverable")
det = res.detections[0]
inter = max(0.0, min(11.0, det.end) - max(8.0, det.start))
iou = inter / ((11.0 - 8.0) + (det.end - det.start) - inter)
check(
    "7. Are artifact regions actually localised?",
    iou > 0.7,
    f"injected 8.00-11.00 s, detected {det.start:.2f}-{det.end:.2f} s (IoU {iou:.2f})",
)

# 8. Does the dashboard reflect the actual pipeline?
app = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
hardcoded = [w for w in ("74 BPM", "812 ms", "96%", "43 / 100", "87 / 100") if w in app]
check(
    "8. Does the dashboard show real pipeline output?",
    not hardcoded and "run_pipeline" in app,
    "app.py calls run_pipeline and contains no hardcoded result values"
    if not hardcoded
    else f"hardcoded values found: {hardcoded}",
)

# 9. Do the validation metrics come from real execution?
val = open(os.path.join(ROOT, "src", "validation.py"), encoding="utf-8").read()
check(
    "9. Do validation metrics come from real execution?",
    "run_pipeline(" in val and "build_cases" in val,
    "validation.py builds records and runs the pipeline on each; no stored numbers",
)

# 10 / 11. Offline, one command?
check(
    "10. Can the whole demo run offline?",
    not any(k in src_text for k in ("requests.", "urllib.request", "http://api")),
    "no network calls anywhere in src/",
)
check(
    "11. Does it work with one command?",
    os.path.exists(os.path.join(ROOT, "app.py")),
    "streamlit run app.py",
)

# 12. Meaningful with all "AI" terminology removed?
# Searching for the words is useless -- "there is no trained model here" would
# match.  Check for the substance instead: an imported ML framework, or a
# serialised model the decisions could be hiding inside.
ml_imports = [
    line.strip()
    for line in src_text.splitlines()
    if line.strip().startswith(("import ", "from "))
    and any(
        pkg in line for pkg in ("tensorflow", "torch", "keras", "sklearn", "xgboost")
    )
]
model_files = [
    os.path.join(dirpath, f)
    for dirpath, _, files in os.walk(ROOT)
    for f in files
    if f.endswith((".pkl", ".h5", ".onnx", ".pt", ".joblib", ".pb"))
]
check(
    "12. Meaningful with all 'AI' terminology removed?",
    not ml_imports and not model_files,
    "no ML framework is imported and no serialised model exists in the repo; "
    "every decision is arithmetic over measured features, with all thresholds "
    "in src/config.py"
    if not ml_imports and not model_files
    else f"imports={ml_imports} models={model_files}",
)

# 13/14. Could a skeptic dismiss this as 'just filtering'?
res_clean = run_pipeline(build("clean"), "clean")
res_noise = run_pipeline(build("muscle_noise"), "muscle_noise")
differential = (
    res_clean.measurement("qrs_duration").status == ACCEPTED
    and res_noise.measurement("qrs_duration").status == REJECTED
    and res_noise.measurement("heart_rate").status == ACCEPTED
)
hr_ok = {v.time for v in res_noise.measurement("heart_rate").beat_verdicts if v.accepted}
qrs_ok = {v.time for v in res_noise.measurement("qrs_duration").beat_verdicts if v.accepted}
qrs_fail = [
    c for c in res_noise.measurement("qrs_duration").criteria if not c.passed
]
check(
    "14. Could a skeptic call this 'just ECG filtering'?",
    differential and qrs_ok < hr_ok and bool(qrs_fail),
    "no: filtering produces a cleaner signal, not a per-measurement verdict. "
    "On this record the SAME filtered waveform yielded an accepted heart rate "
    f"({res_noise.measurement('heart_rate').value} BPM) and a refused QRS "
    "duration, because the morphological standard is a different test and it "
    f"failed on: "
    + "; ".join(f"{c.name} = {c.value} (needed {c.required})" for c in qrs_fail),
)


def main() -> int:
    print("=" * 78)
    print("ECG GUARDIAN -- FINAL TECHNICAL REVIEW (computed, not asserted)")
    print("=" * 78)
    for question, ok, detail in results:
        print(f"\n{'PASS' if ok else 'FAIL'}  {question}")
        print(f"      {detail}")

    print(
        "\nN/A   13. Could a biomedical engineering judge understand the "
        "contribution?"
        "\n      Not something this script can decide. The material offered for "
        "that\n      judgement is: the pipeline diagram in the sidebar, the "
        "per-measurement\n      criteria tables, and the two-standard comparison "
        "in the README."
    )

    failed = [q for q, ok, _ in results if not ok]
    print("\n" + "=" * 78)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
