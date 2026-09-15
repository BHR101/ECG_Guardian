"""The ECG Guardian pipeline.

    ECG -> quality -> artifact detection -> localisation/classification
        -> recovery -> REVALIDATION -> beats -> measurement-specific evidence
        -> accept / reject -> trust map

Each stage records what it did, so the dashboard shows the actual pipeline
rather than a re-telling of it.  No stage is allowed to bring the whole run
down: a failure is caught, logged, and the run continues with whatever
evidence survives -- which, given the design, simply means fewer measurements
are accepted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .config import (
    DISCLAIMER,
    QUALITY_DEGRADED_MIN,
    QUALITY_TRUSTED_MIN,
    TRUST_DEGRADED,
    TRUST_RECOVERED,
    TRUST_REJECTED,
    TRUST_TRUSTED,
)
from .detection import ArtifactDetection, detect_artifacts
from .ecg_generator import ECGRecord
from .evidence import ACCEPTED, MeasurementResult, run_evidence_engine
from .filtering import RecoveryReport, recover
from .measurements import BeatMorphology, measure_morphology
from .peaks import Beat, PeakReport, detect_r_peaks, rescore_beats
from .quality import QualityReport, assess_quality


@dataclass
class TrustSegment:
    """One span of the record with a single trust label."""

    start: float
    end: float
    label: str
    score: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "label": self.label,
            "score": round(self.score, 1),
            "note": self.note,
        }


@dataclass
class LogEntry:
    """One line of the event log."""

    clock: str
    elapsed_ms: float
    stage: str
    message: str
    level: str = "info"

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.clock,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "stage": self.stage,
            "message": self.message,
            "level": self.level,
        }


@dataclass
class PipelineResult:
    """Everything produced by one analysis run."""

    record: ECGRecord
    quality_raw: QualityReport
    detections: list[ArtifactDetection]
    recovery: RecoveryReport
    quality_processed: QualityReport
    peaks: PeakReport
    morphology: list[BeatMorphology]
    measurements: list[MeasurementResult]
    trust_map: list[TrustSegment]
    log: list[LogEntry]
    scenario: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def processed_signal(self) -> np.ndarray:
        return self.recovery.signal

    @property
    def beats(self) -> list[Beat]:
        return self.peaks.beats

    def measurement(self, name: str) -> MeasurementResult | None:
        for m in self.measurements:
            if m.measurement == name:
                return m
        return None

    @property
    def accepted(self) -> list[MeasurementResult]:
        return [m for m in self.measurements if m.status == ACCEPTED]

    @property
    def rejected(self) -> list[MeasurementResult]:
        return [m for m in self.measurements if m.status != ACCEPTED]

    @property
    def overall_trust(self) -> str:
        """A single headline state summarising the whole record."""
        labels = [s.label for s in self.trust_map]
        if TRUST_REJECTED in labels:
            return TRUST_REJECTED
        if TRUST_RECOVERED in labels:
            return TRUST_RECOVERED
        if TRUST_DEGRADED in labels:
            return TRUST_DEGRADED
        return TRUST_TRUSTED

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "disclaimer": DISCLAIMER,
            "quality_raw": round(self.quality_raw.score, 1),
            "quality_processed": round(self.quality_processed.score, 1),
            "recovery_status": self.recovery.status,
            "detections": [d.as_dict() for d in self.detections],
            "beats_detected": len(self.beats),
            "overall_trust": self.overall_trust,
            "measurements": [m.as_dict() for m in self.measurements],
            "trust_map": [s.as_dict() for s in self.trust_map],
            "errors": list(self.errors),
        }


class _Log:
    """Collects timestamped pipeline events."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self._t0 = time.perf_counter()

    def add(self, stage: str, message: str, level: str = "info") -> None:
        self.entries.append(
            LogEntry(
                clock=datetime.now().strftime("%H:%M:%S"),
                elapsed_ms=(time.perf_counter() - self._t0) * 1000.0,
                stage=stage,
                message=message,
                level=level,
            )
        )


def build_trust_map(
    quality: QualityReport,
    recovery: RecoveryReport,
    duration: float,
) -> list[TrustSegment]:
    """Label every part of the record TRUSTED / DEGRADED / RECOVERED / REJECTED.

    Region outcomes from revalidation take precedence over the raw score: a
    stretch whose recovery was validated is RECOVERED even if its score is now
    merely adequate, and a stretch whose recovery failed is REJECTED even if
    filtering happened to raise its score.
    """
    recovered = recovery.recovered_regions
    rejected = recovery.rejected_regions

    def region_label(t: float) -> str | None:
        for s, e in rejected:
            if s <= t < e:
                return TRUST_REJECTED
        for s, e in recovered:
            if s <= t < e:
                return TRUST_RECOVERED
        return None

    raw: list[tuple[float, float, str, float]] = []
    for w in quality.windows:
        centre = 0.5 * (w.start + w.end)
        label = region_label(centre)
        if label is None:
            if w.score >= QUALITY_TRUSTED_MIN:
                label = TRUST_TRUSTED
            elif w.score >= QUALITY_DEGRADED_MIN:
                label = TRUST_DEGRADED
            else:
                label = TRUST_REJECTED
        raw.append((w.start, w.end, label, w.score))

    if not raw:
        return []

    # Merge consecutive windows carrying the same label.
    segments: list[TrustSegment] = []
    cur_start, cur_end, cur_label, scores = raw[0][0], raw[0][1], raw[0][2], [raw[0][3]]
    for start, end, label, score in raw[1:]:
        if label == cur_label:
            cur_end = end
            scores.append(score)
        else:
            segments.append(
                TrustSegment(cur_start, cur_end, cur_label, float(np.mean(scores)))
            )
            cur_start, cur_end, cur_label, scores = start, end, label, [score]
    segments.append(TrustSegment(cur_start, cur_end, cur_label, float(np.mean(scores))))

    # Analysis windows overlap, so adjacent segments would otherwise overlap
    # too.  Put each boundary midway between the two spans that meet there.
    for i in range(len(segments) - 1):
        boundary = 0.5 * (segments[i].end + segments[i + 1].start)
        segments[i].end = boundary
        segments[i + 1].start = boundary

    # Extend the first and last segment to cover the whole record.
    segments[0].start = 0.0
    segments[-1].end = max(duration, segments[-1].end)

    notes = {
        TRUST_TRUSTED: "quality above the trusted threshold",
        TRUST_DEGRADED: "usable for timing, not for morphology",
        TRUST_RECOVERED: "recovery attempted and independently revalidated",
        TRUST_REJECTED: "not trustworthy; excluded from measurements",
    }
    for s in segments:
        s.note = notes.get(s.label, "")
    return segments


def run_pipeline(record: ECGRecord, scenario: str = "") -> PipelineResult:
    """Run the full analysis on one record.

    Args:
        record: the ECG to analyse.  Its ground-truth fields are *not* read by
            any stage of this function.
        scenario: a label carried through to the dashboard.

    Returns:
        A :class:`PipelineResult`.  Always returns; stage failures are recorded
        in ``errors`` and simply reduce the evidence available downstream.
    """
    log = _Log()
    errors: list[str] = []
    fs = record.sampling_rate
    x = np.asarray(record.signal, dtype=float)

    log.add("acquisition", f"ECG loaded: {record.duration:.0f} s at {fs} Hz")

    # -- Stage 1: quality ----------------------------------------------------
    quality_raw = assess_quality(x, fs)
    log.add(
        "quality",
        f"Prototype Signal Quality Index {quality_raw.score:.0f}/100 "
        f"across {len(quality_raw.windows)} windows",
    )

    # -- Stage 2/3: detection, localisation, classification ------------------
    try:
        detections = detect_artifacts(quality_raw, duration=record.duration)
    except Exception as exc:  # pragma: no cover - defensive
        detections = []
        errors.append(f"artifact detection failed: {exc}")
        log.add("detection", f"Artifact detection failed ({exc})", "error")

    if detections:
        for d in detections:
            log.add(
                "detection",
                f"{d.label} detected, {d.start:.1f}-{d.end:.1f} s, "
                f"confidence {d.confidence:.0%}",
                "warn",
            )
            log.add("detection", f"Affected region localised: {d.start:.1f}-{d.end:.1f} s")
    else:
        log.add("detection", "No artifact region found above the detection threshold")

    # -- Stage 4/5: recovery and mandatory revalidation ----------------------
    try:
        recovery = recover(x, fs, detections, quality_raw)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"recovery failed: {exc}")
        log.add("recovery", f"Recovery failed ({exc}); using the raw signal", "error")
        recovery = RecoveryReport(
            signal=x.copy(),
            quality_before=quality_raw,
            quality_after=quality_raw,
            global_before=quality_raw.score,
            global_after=quality_raw.score,
            status="FAILED",
        )

    if detections:
        log.add("recovery", f"Recovery attempted: {recovery.method_label}")
        for r in recovery.regions:
            log.add(
                "revalidation",
                f"{r.start:.1f}-{r.end:.1f} s: {r.score_before:.0f} -> "
                f"{r.score_after:.0f} -- "
                f"{'RECOVERY VALIDATED' if r.recovered else 'RECOVERY FAILED'}",
                "info" if r.recovered else "warn",
            )
        log.add("revalidation", f"Recovery status: {recovery.status}")

    quality_processed = recovery.quality_after or quality_raw

    # -- Stage 6: R-peak detection ------------------------------------------
    try:
        peak_report = detect_r_peaks(recovery.signal, fs)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"R-peak detection failed: {exc}")
        log.add("peaks", f"R-peak detection failed ({exc})", "error")
        peak_report = PeakReport(beats=[], detection_signal=np.zeros_like(x), sampling_rate=fs)

    for beat in peak_report.beats:
        beat.local_quality = quality_processed.score_at(beat.time)

    # Second pass: now that each beat's local quality is known, rebuild the
    # amplitude / prominence / shape references from beats sitting in
    # trustworthy parts of the record, so that a heavily corrupted record does
    # not end up defining "normal" from its own corruption.
    trusted_mask = [
        (not np.isnan(b.local_quality))
        and b.local_quality >= QUALITY_TRUSTED_MIN
        and not any(s <= b.time < e for s, e in recovery.rejected_regions)
        for b in peak_report.beats
    ]
    if any(trusted_mask) and not all(trusted_mask):
        rescore_beats(recovery.signal, fs, peak_report.beats, trusted_mask)
        log.add(
            "peaks",
            f"Beat references rebuilt from {sum(trusted_mask)} beats in "
            f"trusted regions",
        )
    log.add("peaks", f"{len(peak_report.beats)} candidate R-peaks detected")

    # -- Stage 7: morphology -------------------------------------------------
    try:
        morphology = measure_morphology(recovery.signal, fs, peak_report.beats)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"morphology failed: {exc}")
        morphology = [BeatMorphology(beat_time=b.time, reason="morphology unavailable")
                      for b in peak_report.beats]
        log.add("morphology", f"QRS boundary estimation failed ({exc})", "error")

    # -- Stage 8: measurement-specific evidence gating ----------------------
    try:
        measurements = run_evidence_engine(
            peak_report.beats,
            morphology,
            quality_processed,
            rejected_regions=recovery.rejected_regions,
        )
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"evidence engine failed: {exc}")
        measurements = []
        log.add("evidence", f"Evidence engine failed ({exc})", "error")

    for m in measurements:
        if m.status == ACCEPTED:
            log.add(
                "evidence",
                f"{m.display_name} accepted: {m.display_value} "
                f"({m.confidence:.0%} confidence, {m.validated_beats} validated beats)",
            )
        else:
            log.add(
                "evidence",
                f"{m.display_name} rejected -- {m.reason}",
                "warn",
            )

    trust_map = build_trust_map(quality_processed, recovery, record.duration)
    counts: dict[str, int] = {}
    for s in trust_map:
        counts[s.label] = counts.get(s.label, 0) + 1
    log.add(
        "trust_map",
        "Trust map: " + ", ".join(f"{k.lower()} x{v}" for k, v in counts.items()),
    )
    log.add("done", DISCLAIMER)

    return PipelineResult(
        record=record,
        quality_raw=quality_raw,
        detections=detections,
        recovery=recovery,
        quality_processed=quality_processed,
        peaks=peak_report,
        morphology=morphology,
        measurements=measurements,
        trust_map=trust_map,
        log=log.entries,
        scenario=scenario,
        errors=errors,
    )
