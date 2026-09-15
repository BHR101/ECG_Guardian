"""Phase 14 -- prototype validation.

Every number in here is produced by actually running the pipeline over records
whose corruption we injected ourselves.  Nothing is stored, assumed or
hand-written.  Where the system does badly, the table says so.

Ground truth is used *only* here and only after the fact.  The detector, the
quality engine and the evidence engine each receive a waveform and nothing
else.

Run it with::

    python -m src.validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import ARTIFACT_TYPES, corrupt_ecg
from .config import TRUST_REJECTED
from .ecg_generator import ECGRecord, generate_ecg, noise_free_twin
from .evidence import ACCEPTED
from .measurements import measure_morphology
from .peaks import detect_r_peaks
from .pipeline import PipelineResult, run_pipeline

# A detection counts as matching a ground-truth R peak within this tolerance.
PEAK_TOLERANCE_SEC = 0.06

ARTIFACT_START = 8.0
ARTIFACT_DURATION = 3.0
SEVERITIES = (0.4, 0.6, 0.8, 0.95)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def peak_detection_metrics(
    record: ECGRecord, result: PipelineResult
) -> dict[str, float]:
    """Sensitivity, positive predictive value and timing error for R peaks."""
    fs = record.sampling_rate
    t_true = np.asarray(record.r_peaks_true, dtype=float) / fs
    t_det = result.peaks.times

    used: set[int] = set()
    errors: list[float] = []
    for t in t_true:
        if t_det.size == 0:
            break
        d = np.abs(t_det - t)
        order = np.argsort(d)
        for j in order:
            if d[j] > PEAK_TOLERANCE_SEC:
                break
            if j not in used:
                used.add(int(j))
                errors.append((t_det[j] - t) * 1000.0)
                break

    tp = len(used)
    fn = int(t_true.size) - tp
    fp = int(t_det.size) - tp
    return {
        "n_true_beats": int(t_true.size),
        "n_detected": int(t_det.size),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity": tp / t_true.size if t_true.size else float("nan"),
        "ppv": tp / t_det.size if t_det.size else float("nan"),
        "mean_abs_timing_error_ms": float(np.mean(np.abs(errors))) if errors else float("nan"),
    }


def localisation_metrics(
    record: ECGRecord, result: PipelineResult
) -> dict[str, Any]:
    """Overlap and classification accuracy against the injected artifacts."""
    truth = record.metadata.get("artifacts", [])
    if not truth:
        return {
            "injected": 0,
            "iou": float("nan"),
            "classified_correctly": float("nan"),
            "false_alarm_regions": len(result.detections),
        }

    ious, correct = [], []
    for spec in truth:
        t0, t1 = spec["start"], spec["end"]
        best_iou, best_type = 0.0, None
        for d in result.detections:
            inter = max(0.0, min(t1, d.end) - max(t0, d.start))
            union = (t1 - t0) + (d.end - d.start) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_type = iou, d.type
        ious.append(best_iou)
        correct.append(1.0 if best_type == spec["type"] else 0.0)

    return {
        "injected": len(truth),
        "iou": float(np.mean(ious)),
        "classified_correctly": float(np.mean(correct)),
        "false_alarm_regions": max(0, len(result.detections) - len(truth)),
    }


def heart_rate_error(record: ECGRecord, result: PipelineResult) -> dict[str, Any]:
    """Heart-rate error against the generator's true mean rate, when reported."""
    m = result.measurement("heart_rate")
    true_hr = record.metadata.get("true_mean_hr_bpm", float("nan"))
    if m is None or m.status != ACCEPTED or m.value is None:
        return {
            "hr_reported": False,
            "hr_value": None,
            "hr_true": round(true_hr, 2) if not np.isnan(true_hr) else None,
            "hr_abs_error_bpm": float("nan"),
            "hr_confidence": m.confidence if m else float("nan"),
        }
    return {
        "hr_reported": True,
        "hr_value": m.value,
        "hr_true": round(true_hr, 2),
        "hr_abs_error_bpm": abs(m.value - true_hr),
        "hr_confidence": m.confidence,
    }


def qrs_duration_error(
    record: ECGRecord, result: PipelineResult
) -> dict[str, Any]:
    """QRS duration error against a *noise-free reference measured the same way*.

    The reference is the same boundary estimator run on the noise-free twin of
    this record.  Comparing against the generator's nominal Gaussian geometry
    instead would conflate a definitional offset (where a 15%-of-peak-slope
    threshold falls inside a Gaussian tail) with actual measurement error.
    Both numbers are reported so neither is hidden.
    """
    m = result.measurement("qrs_duration")
    nominal = record.metadata.get("true_qrs_duration_ms", float("nan"))
    try:
        twin = noise_free_twin(record)
        tw_peaks = detect_r_peaks(twin.signal, twin.sampling_rate)
        tw_morph = measure_morphology(twin.signal, twin.sampling_rate, tw_peaks.beats)
        durs = [mm.duration_ms for mm in tw_morph if mm.clear]
        reference = float(np.median(durs)) if durs else float("nan")
    except Exception:
        reference = float("nan")

    if m is None or m.status != ACCEPTED or m.value is None:
        return {
            "qrs_reported": False,
            "qrs_value": None,
            "qrs_reference_ms": round(reference, 1) if not np.isnan(reference) else None,
            "qrs_nominal_ms": nominal,
            "qrs_abs_error_ms": float("nan"),
        }
    return {
        "qrs_reported": True,
        "qrs_value": m.value,
        "qrs_reference_ms": round(reference, 1) if not np.isnan(reference) else None,
        "qrs_nominal_ms": nominal,
        "qrs_abs_error_ms": abs(m.value - reference) if not np.isnan(reference) else float("nan"),
    }


def rejection_metrics(record: ECGRecord, result: PipelineResult) -> dict[str, Any]:
    """How much of the injected artifact ends up excluded from measurement."""
    truth = record.metadata.get("artifacts", [])
    if not truth:
        # For a clean record the useful check is the opposite one: how much of
        # it was (wrongly) withheld.
        rejected = sum(
            s.end - s.start for s in result.trust_map if s.label == TRUST_REJECTED
        )
        return {
            "artifact_seconds": 0.0,
            "artifact_seconds_excluded": 0.0,
            "artifact_exclusion_fraction": float("nan"),
            "clean_seconds_rejected": round(rejected, 2),
        }

    total, excluded = 0.0, 0.0
    for spec in truth:
        t0, t1 = spec["start"], spec["end"]
        total += t1 - t0
        for s in result.trust_map:
            if s.label != TRUST_REJECTED:
                continue
            excluded += max(0.0, min(t1, s.end) - max(t0, s.start))

    clean_rejected = 0.0
    for s in result.trust_map:
        if s.label != TRUST_REJECTED:
            continue
        span = s.end - s.start
        overlap = 0.0
        for spec in truth:
            overlap += max(0.0, min(spec["end"], s.end) - max(spec["start"], s.start))
        clean_rejected += max(0.0, span - overlap)

    return {
        "artifact_seconds": round(total, 2),
        "artifact_seconds_excluded": round(min(excluded, total), 2),
        "artifact_exclusion_fraction": round(min(excluded, total) / total, 3) if total else float("nan"),
        "clean_seconds_rejected": round(clean_rejected, 2),
    }


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    """All metrics for one validation case."""

    name: str
    artifact: str
    severity: float
    row: dict[str, Any] = field(default_factory=dict)


def _run_case(name: str, artifact: str, severity: float, record: ECGRecord) -> CaseResult:
    result = run_pipeline(record, name)
    row: dict[str, Any] = {
        "case": name,
        "artifact": artifact,
        "severity": severity,
        "quality_raw": round(result.quality_raw.score, 1),
        "quality_processed": round(result.quality_processed.score, 1),
        "region_quality_before": round(result.recovery.affected_before, 1)
        if not np.isnan(result.recovery.affected_before)
        else None,
        "region_quality_after": round(result.recovery.affected_after, 1)
        if not np.isnan(result.recovery.affected_after)
        else None,
        "recovery_status": result.recovery.status,
        "overall_trust": result.overall_trust,
        "n_accepted_measurements": len(result.accepted),
        "n_rejected_measurements": len(result.rejected),
        "pipeline_errors": len(result.errors),
    }
    row.update(peak_detection_metrics(record, result))
    row.update(localisation_metrics(record, result))
    row.update(heart_rate_error(record, result))
    row.update(qrs_duration_error(record, result))
    row.update(rejection_metrics(record, result))
    return CaseResult(name=name, artifact=artifact, severity=severity, row=row)


def build_cases() -> list[tuple[str, str, float, ECGRecord]]:
    """The validation set: clean records, and every artifact at every severity."""
    cases: list[tuple[str, str, float, ECGRecord]] = []

    for hr in (48, 60, 72, 95, 120):
        cases.append(
            (f"clean HR={hr}", "none", 0.0, generate_ecg(heart_rate=hr, seed=hr))
        )

    base = generate_ecg()
    for artifact in ARTIFACT_TYPES:
        for sev in SEVERITIES:
            cases.append(
                (
                    f"{artifact} s={sev}",
                    artifact,
                    sev,
                    corrupt_ecg(
                        base,
                        artifact,
                        start=ARTIFACT_START,
                        duration=ARTIFACT_DURATION,
                        severity=sev,
                        seed=7,
                    ),
                )
            )

    # Record-wide degradation.  These are the cases where every beat is
    # affected, so there is no clean remainder to fall back on -- which is
    # where measurement-specific gating has to do its work.
    for artifact in ("muscle_noise", "powerline", "motion"):
        for sev in (0.6, 0.9):
            cases.append(
                (
                    f"{artifact} global s={sev}",
                    artifact,
                    sev,
                    corrupt_ecg(base, artifact, severity=sev, seed=9),
                )
            )
    return cases


def run_validation() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the whole validation set and summarise it.

    Returns:
        ``(per_case_table, summary)``.  The summary holds the headline metrics
        the dashboard shows; the table holds every case behind them.
    """
    rows = [_run_case(*case).row for case in build_cases()]
    df = pd.DataFrame(rows)

    clean = df[df["artifact"] == "none"]
    corrupt = df[df["artifact"] != "none"]
    # The meaningful exclusion question is not "was it severe" but "did
    # recovery fail" -- a severe artifact that was genuinely repaired should
    # NOT be excluded, and counting it as a miss would be wrong.
    not_recovered = corrupt[corrupt["recovery_status"].isin(["FAILED", "PARTIAL"])]
    # Contact loss below severity 0.6 is mostly a baseline step with the signal
    # still largely present, and is classified as wander; restrict the "was
    # unrecoverable damage wrongly declared repaired" check to the cases where
    # contact loss is unambiguous.
    unrecoverable = corrupt[
        (corrupt["artifact"] == "electrode_contact") & (corrupt["severity"] >= 0.6)
    ]
    separable = corrupt[
        corrupt["artifact"].isin(["baseline_wander", "powerline", "muscle_noise"])
    ]

    def _mean(frame: pd.DataFrame, col: str) -> float:
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else float("nan")

    hr_reported = df[df["hr_reported"]]

    summary: dict[str, Any] = {
        "n_cases": int(len(df)),
        # 1. R-peak detection
        "peak_sensitivity": _mean(df, "sensitivity"),
        "peak_ppv": _mean(df, "ppv"),
        "peak_timing_error_ms": _mean(df, "mean_abs_timing_error_ms"),
        # 2/3. Artifact localisation and classification
        "localisation_iou": _mean(corrupt, "iou"),
        "classification_accuracy": _mean(corrupt, "classified_correctly"),
        "false_alarm_regions_on_clean": int(
            pd.to_numeric(clean["false_alarm_regions"], errors="coerce").fillna(0).sum()
        ),
        # 4. Quality score behaviour
        "clean_quality_mean": _mean(clean, "quality_raw"),
        "corrupt_region_quality_mean": _mean(corrupt, "region_quality_before"),
        "clean_seconds_wrongly_rejected": _mean(clean, "clean_seconds_rejected"),
        # 5. Recovery
        "separable_recovery_validated_rate": float(
            (separable["recovery_status"] == "VALIDATED").mean()
        )
        if len(separable)
        else float("nan"),
        "separable_region_quality_gain": _mean(separable, "region_quality_after")
        - _mean(separable, "region_quality_before"),
        "contact_loss_recovery_validated_rate": float(
            (unrecoverable["recovery_status"] == "VALIDATED").mean()
        )
        if len(unrecoverable)
        else float("nan"),
        # 6. Heart rate
        "hr_reported_rate": float(df["hr_reported"].mean()),
        "hr_abs_error_bpm": _mean(hr_reported, "hr_abs_error_bpm"),
        "hr_max_abs_error_bpm": float(
            pd.to_numeric(hr_reported["hr_abs_error_bpm"], errors="coerce").max()
        )
        if len(hr_reported)
        else float("nan"),
        # QRS duration
        "qrs_reported_rate": float(df["qrs_reported"].mean()),
        "qrs_abs_error_ms": _mean(df[df["qrs_reported"]], "qrs_abs_error_ms"),
        # 7. Rejection of corrupted segments
        "failed_recovery_exclusion_fraction": _mean(
            not_recovered, "artifact_exclusion_fraction"
        ),
        "n_failed_recovery_cases": int(len(not_recovered)),
        "measurements_accepted_when_recovery_failed": _mean(
            not_recovered, "n_accepted_measurements"
        ),
        "pipeline_errors": int(
            pd.to_numeric(df["pipeline_errors"], errors="coerce").fillna(0).sum()
        ),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Extra structural checks
# ---------------------------------------------------------------------------
def differential_gating_check() -> dict[str, Any]:
    """Confirm the system really can reach different verdicts per measurement.

    A single global quality gate could not produce this result, so it is the
    check that most directly tests the project's claim.
    """
    from .demo import build

    record = build("muscle_noise")
    result = run_pipeline(record, "muscle_noise")
    hr = result.measurement("heart_rate")
    qrs = result.measurement("qrs_duration")
    return {
        "scenario": "Muscle noise (mild degradation)",
        "heart_rate_status": hr.status if hr else "missing",
        "heart_rate_value": hr.value if hr else None,
        "qrs_status": qrs.status if qrs else "missing",
        "qrs_reason": qrs.reason if qrs else "",
        "different_verdicts": bool(hr and qrs and hr.status != qrs.status),
    }


def ground_truth_isolation_check() -> dict[str, Any]:
    """Confirm no analysis entry point can even see the ground truth.

    Checked by inspecting the signatures of the analysis functions: each takes
    a waveform and a sampling rate, so there is no channel through which the
    true peak positions could reach them.
    """
    import inspect

    from . import detection, evidence, peaks, quality

    checks = {
        "detect_r_peaks": list(inspect.signature(peaks.detect_r_peaks).parameters),
        "assess_quality": list(inspect.signature(quality.assess_quality).parameters),
        "detect_artifacts": list(
            inspect.signature(detection.detect_artifacts).parameters
        ),
        "run_evidence_engine": list(
            inspect.signature(evidence.run_evidence_engine).parameters
        ),
    }
    banned = ("r_peaks_true", "ground_truth", "truth", "record")
    leaks = {
        name: [p for p in params if any(b in p for b in banned)]
        for name, params in checks.items()
    }
    return {
        "signatures": checks,
        "leaks": {k: v for k, v in leaks.items() if v},
        "clean": all(not v for v in leaks.values()),
    }


def adversarial_summary(n_random: int = 60) -> dict[str, Any]:
    """Run the adversarial corpus and return the safety-critical metrics.

    The headline number is the FALSE ACCEPTANCE RATE: how often the system
    reported a measurement whose value was materially wrong against ground
    truth.  Conservative rejection is counted separately and is not treated as
    a failure -- refusing to answer is a valid answer here.
    """
    from .adversarial import FALSE_ACCEPT, FALSE_REJECT, TRUE_ACCEPT, all_cases, judge_case

    rows, crashes = [], 0
    for case in all_cases(include_random=n_random):
        try:
            result = run_pipeline(case.record, case.key)
        except Exception:
            crashes += 1
            continue
        rows.append(judge_case(case, result))

    df = pd.DataFrame(rows)
    out: dict[str, Any] = {
        "n_cases": len(df),
        "n_random": n_random,
        "crashes": crashes,
        "stage_errors": int(pd.to_numeric(df["pipeline_errors"], errors="coerce").fillna(0).sum()),
    }
    total_reported = total_false = 0
    for key, label in (("hr", "heart_rate"), ("qrs", "qrs_duration")):
        outcome = df[f"{key}_outcome"]
        reported = int(df[f"{key}_reported"].fillna(False).sum())
        false_acc = int((outcome == FALSE_ACCEPT).sum())
        conf = pd.to_numeric(
            df.loc[outcome == FALSE_ACCEPT, f"{key}_confidence"], errors="coerce"
        )
        out[f"{label}_reported"] = reported
        out[f"{label}_true_accept"] = int((outcome == TRUE_ACCEPT).sum())
        out[f"{label}_false_accept"] = false_acc
        out[f"{label}_false_reject"] = int((outcome == FALSE_REJECT).sum())
        out[f"{label}_false_accept_rate"] = false_acc / reported if reported else 0.0
        # P0 = wrong number reported confidently; P1 = wrong number, low confidence.
        out[f"{label}_p0"] = int((conf >= 0.80).sum()) if false_acc else 0
        out[f"{label}_p1"] = false_acc - out[f"{label}_p0"]
        total_reported += reported
        total_false += false_acc

    out["overall_reported"] = total_reported
    out["overall_false_accept"] = total_false
    out["false_acceptance_rate"] = total_false / total_reported if total_reported else 0.0
    out["p0_count"] = out["heart_rate_p0"] + out["qrs_duration_p0"]
    out["p1_count"] = out["heart_rate_p1"] + out["qrs_duration_p1"]
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import warnings

    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    df, summary = run_validation()

    print("=" * 78)
    print("ECG GUARDIAN -- PROTOTYPE VALIDATION")
    print("Prototype engineering system. Not intended for clinical diagnosis.")
    print("=" * 78)
    print(f"\nCases run: {summary['n_cases']}\n")

    groups = [
        ("1. R-peak detection", [
            ("sensitivity", "peak_sensitivity", "{:.3f}"),
            ("positive predictive value", "peak_ppv", "{:.3f}"),
            ("mean |timing error|", "peak_timing_error_ms", "{:.2f} ms"),
        ]),
        ("2/3. Artifact localisation and classification", [
            ("mean IoU with injected region", "localisation_iou", "{:.3f}"),
            ("classification accuracy", "classification_accuracy", "{:.3f}"),
            ("false-alarm regions on clean records", "false_alarm_regions_on_clean", "{:.0f}"),
        ]),
        ("4. Quality score behaviour", [
            ("mean score, clean records", "clean_quality_mean", "{:.1f}/100"),
            ("mean score in corrupted regions", "corrupt_region_quality_mean", "{:.1f}/100"),
            ("clean seconds wrongly rejected", "clean_seconds_wrongly_rejected", "{:.2f} s"),
        ]),
        ("5. Recovery and revalidation", [
            ("separable artifacts revalidated", "separable_recovery_validated_rate", "{:.1%}"),
            ("their mean region quality gain", "separable_region_quality_gain", "+{:.1f} points"),
            ("contact loss (sev>=0.6) wrongly revalidated", "contact_loss_recovery_validated_rate", "{:.1%}"),
        ]),
        ("6. Heart rate", [
            ("cases where HR was reported", "hr_reported_rate", "{:.1%}"),
            ("mean |error| when reported", "hr_abs_error_bpm", "{:.2f} BPM"),
            ("worst |error| when reported", "hr_max_abs_error_bpm", "{:.2f} BPM"),
        ]),
        ("7. QRS duration", [
            ("cases where QRS was reported", "qrs_reported_rate", "{:.1%}"),
            ("mean |error| vs noise-free reference", "qrs_abs_error_ms", "{:.2f} ms"),
        ]),
        ("8. Rejection of corrupted segments", [
            ("cases where recovery failed", "n_failed_recovery_cases", "{:.0f}"),
            ("their artifact time excluded", "failed_recovery_exclusion_fraction", "{:.1%}"),
            ("measurements still accepted there", "measurements_accepted_when_recovery_failed", "{:.1f} of 4"),
            ("uncaught pipeline errors", "pipeline_errors", "{:.0f}"),
        ]),
    ]

    for title, items in groups:
        print(title)
        for label, key, fmt in items:
            value = summary.get(key, float("nan"))
            try:
                shown = fmt.format(value)
            except (TypeError, ValueError):
                shown = str(value)
            print(f"    {label:38s} {shown}")
        print()

    gate = differential_gating_check()
    print("9. Measurement-specific gating (the central claim)")
    print(f"    scenario: {gate['scenario']}")
    print(f"    heart rate  -> {gate['heart_rate_status']} ({gate['heart_rate_value']} BPM)")
    print(f"    QRS duration-> {gate['qrs_status']}")
    print(f"    reason: {gate['qrs_reason'][:90]}")
    print(f"    different verdicts from one record: {gate['different_verdicts']}\n")

    iso = ground_truth_isolation_check()
    print("10. Ground-truth isolation")
    for name, params in iso["signatures"].items():
        print(f"    {name}({', '.join(params)})")
    print(f"    no ground-truth parameter reaches any analysis stage: {iso['clean']}\n")

    print("11. Adversarial corpus -- the safety-critical metric")
    adv = adversarial_summary()
    print(f"    cases run                              {adv['n_cases']} "
          f"({adv['n_random']} randomised)")
    print(f"    crashes / uncaught stage errors        {adv['crashes']} / {adv['stage_errors']}")
    for label, name in (("heart_rate", "heart rate"), ("qrs_duration", "QRS duration")):
        print(f"    {name:14s} reported            {adv[label + '_reported']}")
        print(f"    {name:14s} FALSE ACCEPTANCES   {adv[label + '_false_accept']}")
        print(f"    {name:14s} conservative refusals {adv[label + '_false_reject']}")
    print(f"    P0 (wrong value, confidence >= 0.80)   {adv['p0_count']}")
    print(f"    P1 (wrong value, low confidence)       {adv['p1_count']}")
    print(f"    FALSE ACCEPTANCE RATE                  "
          f"{adv['false_acceptance_rate']:.2%} of all reported values")
    print()

    print("Per-case detail")
    cols = [
        "case", "quality_raw", "region_quality_before", "region_quality_after",
        "recovery_status", "sensitivity", "ppv", "iou", "classified_correctly",
        "hr_reported", "hr_abs_error_bpm", "qrs_reported",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
