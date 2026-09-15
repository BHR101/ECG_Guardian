"""Phase 9 -- measurement-specific evidence gating.

This is the module the project exists for.

Conventional pipelines compute one signal-quality number and apply it as a
single gate to everything downstream.  That throws away a real distinction: a
moderately degraded ECG can easily contain enough evidence to say *when* the
beats happened while containing nothing like enough to say *how wide* the
complexes were.

So here, each measurement declares what evidence it needs, that evidence is
counted beat by beat, and the measurement is accepted or rejected on its own
terms.  Two measurements derived from the same record routinely get different
answers.  Where the evidence is insufficient, no number is produced -- the
measurement is refused, with the failing criterion named.

Every confidence value is a weighted combination of measured quantities.  None
is drawn at random.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .config import (
    BEAT_STANDARD_MORPHOLOGY,
    BEAT_STANDARD_TIMING,
    HR_MIN_CONFIDENCE,
    HR_MIN_LOCAL_QUALITY,
    HR_MIN_RR_CONSISTENCY,
    HR_MIN_VALID_BEAT_FRACTION,
    HR_MIN_VALID_BEATS,
    QRS_MAX_DISPERSION_MS,
    QRS_MIN_BEAT_FRACTION,
    QRS_MIN_CONFIDENCE,
    QRS_MIN_LOCAL_QUALITY,
    QRS_MIN_VALID_BEATS,
    QRS_PLAUSIBLE_MS,
    RR_MIN_CONFIDENCE,
    RR_MIN_CONSISTENCY,
    RR_MIN_VALID_INTERVALS,
    RRC_MIN_CONFIDENCE,
    RRC_MIN_VALID_INTERVALS,
)
from .measurements import BeatMorphology, rr_from_beats
from .peaks import Beat
from .quality import QualityReport

EPS = 1e-12

ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Beat-level evidence
# ---------------------------------------------------------------------------
@dataclass
class BeatVerdict:
    """Whether one beat meets one measurement's evidence standard."""

    time: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    local_quality: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "local_quality": round(self.local_quality, 1)
            if not np.isnan(self.local_quality)
            else None,
        }


def _in_regions(t: float, regions: Sequence[tuple[float, float]]) -> bool:
    return any(s <= t < e for s, e in regions)


def judge_beats(
    beats: list[Beat],
    standard: dict[str, Any],
    rejected_regions: Sequence[tuple[float, float]] = (),
    morphology: Sequence[BeatMorphology] | None = None,
    hf_noise_at: Sequence[float] | None = None,
) -> list[BeatVerdict]:
    """Judge every beat against one evidence standard.

    Args:
        beats: detected beats, already annotated with ``local_quality``.
        standard: one of the ``BEAT_STANDARD_*`` dicts from :mod:`src.config`.
        rejected_regions: spans whose recovery failed revalidation.  Beats
            inside these are refused regardless of how they look, because the
            pipeline has already established that the region is untrustworthy.
        morphology: per-beat QRS boundary results, required when the standard
            includes morphological criteria.
        hf_noise_at: per-beat local high-frequency noise ratio.

    Returns:
        One :class:`BeatVerdict` per beat, in order.
    """
    lo_amp, hi_amp = standard["amplitude_range"]
    verdicts: list[BeatVerdict] = []

    for i, b in enumerate(beats):
        reasons: list[str] = []

        if _in_regions(b.time, rejected_regions):
            reasons.append("inside a region whose recovery failed revalidation")

        q = b.local_quality
        if np.isnan(q) or q < standard["min_local_quality"]:
            reasons.append(
                f"local signal quality {q:.0f} < {standard['min_local_quality']:.0f}"
            )
        if b.relative_prominence < standard["min_relative_prominence"]:
            reasons.append(
                f"relative R-peak prominence {b.relative_prominence:.2f} < "
                f"{standard['min_relative_prominence']:.2f}"
            )
        if not (lo_amp <= b.relative_amplitude <= hi_amp):
            reasons.append(
                f"relative amplitude {b.relative_amplitude:.2f} outside "
                f"[{lo_amp:.2f}, {hi_amp:.2f}]"
            )
        if b.template_correlation < standard["min_template_correlation"]:
            reasons.append(
                f"beat shape correlation {b.template_correlation:.2f} < "
                f"{standard['min_template_correlation']:.2f}"
            )

        if "max_hf_noise_ratio" in standard and hf_noise_at is not None:
            hf = hf_noise_at[i] if i < len(hf_noise_at) else float("nan")
            if np.isnan(hf) or hf > standard["max_hf_noise_ratio"]:
                reasons.append(
                    f"local high-frequency noise {hf:.2f} > "
                    f"{standard['max_hf_noise_ratio']:.2f}"
                )

        if "min_slope_contrast" in standard:
            morph = morphology[i] if morphology is not None and i < len(morphology) else None
            if morph is None or not morph.clear:
                reasons.append(
                    morph.reason if morph is not None else "no morphology estimate"
                )
            elif morph.slope_contrast < standard["min_slope_contrast"]:
                reasons.append(
                    f"QRS slope contrast {morph.slope_contrast:.1f} < "
                    f"{standard['min_slope_contrast']:.1f}"
                )

        verdicts.append(
            BeatVerdict(
                time=b.time,
                accepted=not reasons,
                reasons=reasons,
                local_quality=q,
            )
        )
    return verdicts


def _spans_from_beats(
    beats: list[Beat], verdicts: list[BeatVerdict], accepted: bool
) -> list[tuple[float, float]]:
    """Group runs of consecutive (non-)accepted beats into time spans."""
    spans: list[list[float]] = []
    for b, v in zip(beats, verdicts):
        if v.accepted != accepted:
            continue
        if spans and b.time - spans[-1][1] < 2.0:
            spans[-1][1] = b.time
        else:
            spans.append([b.time, b.time])
    return [(round(a, 2), round(bb, 2)) for a, bb in spans]


# ---------------------------------------------------------------------------
# Measurement results
# ---------------------------------------------------------------------------
@dataclass
class Criterion:
    """One named requirement, its measured value, and whether it was met."""

    name: str
    value: float | int | str
    required: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        v = self.value
        if isinstance(v, float):
            v = round(v, 3)
        return {"name": self.name, "value": v, "required": self.required, "passed": self.passed}


@dataclass
class MeasurementResult:
    """A measurement together with the full record of why it was or was not reported."""

    measurement: str
    display_name: str
    unit: str = ""
    value: float | None = None
    confidence: float = 0.0
    status: str = REJECTED
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    criteria: list[Criterion] = field(default_factory=list)
    validated_beats: int = 0
    rejected_beats: int = 0
    supporting_segments: list[tuple[float, float]] = field(default_factory=list)
    rejected_segments: list[tuple[float, float]] = field(default_factory=list)
    beat_verdicts: list[BeatVerdict] = field(default_factory=list)
    evidence_standard: str = ""
    confidence_terms: dict[str, float] = field(default_factory=dict)

    @property
    def display_value(self) -> str:
        if self.status != ACCEPTED or self.value is None:
            return "NOT REPORTED"
        return f"{self.value:g} {self.unit}".strip()

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["criteria"] = [c.as_dict() for c in self.criteria]
        d["beat_verdicts"] = [v.as_dict() for v in self.beat_verdicts]
        d["display_value"] = self.display_value
        return d


def _finalise(
    result: MeasurementResult,
    value: float | None,
    confidence: float,
    min_confidence: float,
) -> MeasurementResult:
    """Apply the criteria list and the confidence floor to reach a verdict."""
    failed = [c for c in result.criteria if not c.passed]
    conf_ok = confidence >= min_confidence
    result.criteria.append(
        Criterion(
            name="measurement confidence",
            value=round(confidence, 3),
            required=f">= {min_confidence:.2f}",
            passed=conf_ok,
        )
    )
    result.confidence = round(float(np.clip(confidence, 0.0, 0.99)), 3)

    if failed or not conf_ok:
        result.status = REJECTED
        result.value = None
        if failed:
            result.reason = "; ".join(
                f"{c.name} was {c.value} (required {c.required})" for c in failed[:3]
            )
        else:
            result.reason = (
                f"measurement confidence {confidence:.2f} below the required "
                f"{min_confidence:.2f}"
            )
    else:
        result.status = ACCEPTED
        result.value = value
        result.reason = ""
    return result


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------
def gate_heart_rate(
    beats: list[Beat], verdicts: list[BeatVerdict]
) -> MeasurementResult:
    """Decide whether heart rate has enough supporting evidence.

    Heart rate needs beats whose *timing* is reliable: enough of them, agreeing
    with each other, in parts of the record whose local quality is adequate.
    It does not need clean morphology.
    """
    res = MeasurementResult(
        measurement="heart_rate",
        display_name="Heart rate",
        unit="BPM",
        evidence_standard=str(BEAT_STANDARD_TIMING["label"]),
        beat_verdicts=verdicts,
    )
    valid = [b for b, v in zip(beats, verdicts) if v.accepted]
    res.validated_beats = len(valid)
    res.rejected_beats = len(beats) - len(valid)
    res.supporting_segments = _spans_from_beats(beats, verdicts, True)
    res.rejected_segments = _spans_from_beats(beats, verdicts, False)

    rhythm = rr_from_beats(valid)
    n_valid = len(valid)
    fraction = n_valid / max(1, len(beats))
    consistency = rhythm.consistency if not np.isnan(rhythm.consistency) else 0.0
    med_q = (
        float(np.median([b.local_quality for b in valid]))
        if valid
        else float("nan")
    )
    med_q_safe = 0.0 if np.isnan(med_q) else med_q

    res.criteria = [
        Criterion("validated beats", n_valid, f">= {HR_MIN_VALID_BEATS}", n_valid >= HR_MIN_VALID_BEATS),
        Criterion(
            "validated beat fraction", round(fraction, 3),
            f">= {HR_MIN_VALID_BEAT_FRACTION:.2f}", fraction >= HR_MIN_VALID_BEAT_FRACTION,
        ),
        Criterion(
            "usable RR intervals", rhythm.n_intervals,
            ">= 4", rhythm.n_intervals >= 4,
        ),
        Criterion(
            "RR consistency", round(consistency, 3),
            f">= {HR_MIN_RR_CONSISTENCY:.2f}", consistency >= HR_MIN_RR_CONSISTENCY,
        ),
        Criterion(
            "median local quality at validated beats", round(med_q_safe, 1),
            f">= {HR_MIN_LOCAL_QUALITY:.0f}", med_q_safe >= HR_MIN_LOCAL_QUALITY,
        ),
    ]

    quality_factor = float(np.clip((med_q_safe - 50.0) / 40.0, 0.0, 1.0))
    terms = {
        "validated beat fraction": 0.40 * fraction,
        "RR consistency": 0.35 * consistency,
        "local quality": 0.25 * quality_factor,
    }
    confidence = float(sum(terms.values()))
    res.confidence_terms = {k: round(v, 3) for k, v in terms.items()}

    res.evidence = [
        f"{n_valid} of {len(beats)} detected beats met the timing standard",
        f"{rhythm.n_intervals} usable RR intervals",
        f"RR consistency {consistency * 100:.0f}% "
        f"(intervals within 12.5% of the median)",
        f"median local signal quality {med_q_safe:.0f}/100 at the validated beats",
    ]
    if rhythm.n_excluded_intervals:
        res.evidence.append(
            f"{rhythm.n_excluded_intervals} interval(s) discarded because a beat "
            f"between them was rejected"
        )

    value = round(rhythm.heart_rate, 1) if not np.isnan(rhythm.heart_rate) else None
    if value is None:
        res.criteria.append(
            Criterion("heart rate computable", "no", "yes", False)
        )
    return _finalise(res, value, confidence, HR_MIN_CONFIDENCE)


def gate_rr_interval(
    beats: list[Beat], verdicts: list[BeatVerdict]
) -> MeasurementResult:
    """Decide whether a representative RR interval may be reported."""
    res = MeasurementResult(
        measurement="rr_interval",
        display_name="RR interval",
        unit="ms",
        evidence_standard=str(BEAT_STANDARD_TIMING["label"]),
        beat_verdicts=verdicts,
    )
    valid = [b for b, v in zip(beats, verdicts) if v.accepted]
    res.validated_beats = len(valid)
    res.rejected_beats = len(beats) - len(valid)
    res.supporting_segments = _spans_from_beats(beats, verdicts, True)
    res.rejected_segments = _spans_from_beats(beats, verdicts, False)

    rhythm = rr_from_beats(valid)
    consistency = rhythm.consistency if not np.isnan(rhythm.consistency) else 0.0
    fraction = len(valid) / max(1, len(beats))
    med_q = (
        float(np.median([b.local_quality for b in valid])) if valid else 0.0
    )
    med_q = 0.0 if np.isnan(med_q) else med_q

    res.criteria = [
        Criterion(
            "usable RR intervals", rhythm.n_intervals,
            f">= {RR_MIN_VALID_INTERVALS}", rhythm.n_intervals >= RR_MIN_VALID_INTERVALS,
        ),
        Criterion(
            "RR consistency", round(consistency, 3),
            f">= {RR_MIN_CONSISTENCY:.2f}", consistency >= RR_MIN_CONSISTENCY,
        ),
        Criterion(
            "validated beat fraction", round(fraction, 3),
            f">= {HR_MIN_VALID_BEAT_FRACTION:.2f}", fraction >= HR_MIN_VALID_BEAT_FRACTION,
        ),
    ]

    quality_factor = float(np.clip((med_q - 50.0) / 40.0, 0.0, 1.0))
    terms = {
        "validated beat fraction": 0.30 * fraction,
        "RR consistency": 0.45 * consistency,
        "local quality": 0.25 * quality_factor,
    }
    confidence = float(sum(terms.values()))
    res.confidence_terms = {k: round(v, 3) for k, v in terms.items()}

    res.evidence = [
        f"{rhythm.n_intervals} RR intervals between consecutively validated beats",
        f"RR consistency {consistency * 100:.0f}%",
        f"interquartile spread {rhythm.robust_cv * 100:.1f}% of the median"
        if not np.isnan(rhythm.robust_cv)
        else "interquartile spread not computable",
    ]

    value = round(rhythm.median_rr * 1000.0, 1) if not np.isnan(rhythm.median_rr) else None
    if value is None:
        res.criteria.append(Criterion("RR interval computable", "no", "yes", False))
    return _finalise(res, value, confidence, RR_MIN_CONFIDENCE)


def gate_rr_consistency(
    beats: list[Beat], verdicts: list[BeatVerdict]
) -> MeasurementResult:
    """Report RR consistency itself as a gated measurement."""
    res = MeasurementResult(
        measurement="rr_consistency",
        display_name="RR consistency",
        unit="%",
        evidence_standard=str(BEAT_STANDARD_TIMING["label"]),
        beat_verdicts=verdicts,
    )
    valid = [b for b, v in zip(beats, verdicts) if v.accepted]
    res.validated_beats = len(valid)
    res.rejected_beats = len(beats) - len(valid)
    res.supporting_segments = _spans_from_beats(beats, verdicts, True)
    res.rejected_segments = _spans_from_beats(beats, verdicts, False)

    rhythm = rr_from_beats(valid)
    consistency = rhythm.consistency if not np.isnan(rhythm.consistency) else 0.0
    fraction = len(valid) / max(1, len(beats))
    med_q = float(np.median([b.local_quality for b in valid])) if valid else 0.0
    med_q = 0.0 if np.isnan(med_q) else med_q

    res.criteria = [
        Criterion(
            "usable RR intervals", rhythm.n_intervals,
            f">= {RRC_MIN_VALID_INTERVALS}", rhythm.n_intervals >= RRC_MIN_VALID_INTERVALS,
        ),
        Criterion(
            "validated beat fraction", round(fraction, 3),
            f">= {HR_MIN_VALID_BEAT_FRACTION:.2f}", fraction >= HR_MIN_VALID_BEAT_FRACTION,
        ),
    ]

    sample_factor = float(np.clip(rhythm.n_intervals / 20.0, 0.0, 1.0))
    quality_factor = float(np.clip((med_q - 50.0) / 40.0, 0.0, 1.0))
    terms = {
        "validated beat fraction": 0.50 * fraction,
        "sample size": 0.30 * sample_factor,
        "local quality": 0.20 * quality_factor,
    }
    confidence = float(sum(terms.values()))
    res.confidence_terms = {k: round(v, 3) for k, v in terms.items()}
    res.evidence = [
        f"computed over {rhythm.n_intervals} RR intervals",
        f"{res.validated_beats} validated beats, {res.rejected_beats} rejected",
    ]

    value = round(consistency * 100.0, 1) if rhythm.n_intervals else None
    if value is None:
        res.criteria.append(Criterion("consistency computable", "no", "yes", False))
    return _finalise(res, value, confidence, RRC_MIN_CONFIDENCE)


def gate_qrs_duration(
    beats: list[Beat],
    verdicts: list[BeatVerdict],
    morphology: Sequence[BeatMorphology],
) -> MeasurementResult:
    """Decide whether QRS duration has enough *morphological* evidence.

    This gate is deliberately stricter than the rhythm gates and consumes
    different evidence: it needs resolved onsets and offsets that agree with
    each other, not merely reliably timed peaks.  On a degraded record it is
    expected to refuse where heart rate is accepted.
    """
    res = MeasurementResult(
        measurement="qrs_duration",
        display_name="QRS duration",
        unit="ms",
        evidence_standard=str(BEAT_STANDARD_MORPHOLOGY["label"]),
        beat_verdicts=verdicts,
    )
    pairs = [
        (b, m)
        for b, v, m in zip(beats, verdicts, morphology)
        if v.accepted and m.clear and not np.isnan(m.duration_ms)
    ]
    res.validated_beats = len(pairs)
    res.rejected_beats = len(beats) - len(pairs)
    res.supporting_segments = _spans_from_beats(beats, verdicts, True)
    res.rejected_segments = _spans_from_beats(beats, verdicts, False)

    durations = np.asarray([m.duration_ms for _, m in pairs], dtype=float)
    fraction = len(pairs) / max(1, len(beats))
    med_dur = float(np.median(durations)) if durations.size else float("nan")
    iqr_ms = (
        float(np.percentile(durations, 75) - np.percentile(durations, 25))
        if durations.size >= 4
        else float("nan")
    )
    med_q = (
        float(np.median([b.local_quality for b, _ in pairs])) if pairs else 0.0
    )
    med_q = 0.0 if np.isnan(med_q) else med_q
    med_corr = (
        float(np.median([b.template_correlation for b, _ in pairs])) if pairs else 0.0
    )
    plaus_lo, plaus_hi = QRS_PLAUSIBLE_MS

    res.criteria = [
        Criterion(
            "beats with resolved QRS onset and offset", len(pairs),
            f">= {QRS_MIN_VALID_BEATS}", len(pairs) >= QRS_MIN_VALID_BEATS,
        ),
        Criterion(
            "fraction of beats with usable morphology", round(fraction, 3),
            f">= {QRS_MIN_BEAT_FRACTION:.2f}", fraction >= QRS_MIN_BEAT_FRACTION,
        ),
        Criterion(
            "median local quality at those beats", round(med_q, 1),
            f">= {QRS_MIN_LOCAL_QUALITY:.0f}", med_q >= QRS_MIN_LOCAL_QUALITY,
        ),
        Criterion(
            "beat-to-beat spread of QRS duration (IQR)",
            round(iqr_ms, 1) if not np.isnan(iqr_ms) else "not computable",
            f"<= {QRS_MAX_DISPERSION_MS:.0f} ms",
            (not np.isnan(iqr_ms)) and iqr_ms <= QRS_MAX_DISPERSION_MS,
        ),
        Criterion(
            "median QRS duration within plausible range",
            round(med_dur, 1) if not np.isnan(med_dur) else "not computable",
            f"{plaus_lo:.0f}-{plaus_hi:.0f} ms",
            (not np.isnan(med_dur)) and plaus_lo <= med_dur <= plaus_hi,
        ),
    ]

    morph_fraction = fraction
    dispersion_factor = (
        float(np.clip(1.0 - iqr_ms / 20.0, 0.0, 1.0)) if not np.isnan(iqr_ms) else 0.0
    )
    quality_factor = float(np.clip((med_q - 60.0) / 35.0, 0.0, 1.0))
    template_factor = float(np.clip((med_corr - 0.85) / 0.13, 0.0, 1.0))
    terms = {
        "beats with usable morphology": 0.30 * morph_fraction,
        "onset/offset agreement": 0.25 * dispersion_factor,
        "local quality": 0.25 * quality_factor,
        "beat shape consistency": 0.20 * template_factor,
    }
    confidence = float(sum(terms.values()))
    res.confidence_terms = {k: round(v, 3) for k, v in terms.items()}

    if pairs:
        res.evidence = [
            f"{len(pairs)} of {len(beats)} beats gave a resolved QRS onset and offset",
            f"beat-to-beat spread (IQR) {iqr_ms:.1f} ms"
            if not np.isnan(iqr_ms)
            else "beat-to-beat spread not computable",
            f"median beat-shape correlation {med_corr:.2f}",
            f"median local signal quality {med_q:.0f}/100",
        ]
    else:
        common = _dominant_reason(verdicts, morphology)
        res.evidence = [
            "no beat met the morphological evidence standard",
            f"most common limitation: {common}" if common else "",
        ]
        res.evidence = [e for e in res.evidence if e]

    value = round(med_dur, 1) if not np.isnan(med_dur) else None
    out = _finalise(res, value, confidence, QRS_MIN_CONFIDENCE)
    if out.status == REJECTED and not pairs:
        out.reason = "Insufficient morphological evidence: " + out.reason
    return out


def _dominant_reason(
    verdicts: Sequence[BeatVerdict], morphology: Sequence[BeatMorphology]
) -> str:
    """The most frequently cited reason beats failed, for the rejection message."""
    counts: dict[str, int] = {}
    for v in verdicts:
        for r in v.reasons:
            key = r.split(" (")[0]
            # Collapse the numeric part so similar reasons group together.
            key = key.split(" <")[0].split(" >")[0].split(" outside")[0]
            counts[key] = counts.get(key, 0) + 1
    for m in morphology:
        if not m.clear and m.reason:
            counts[m.reason] = counts.get(m.reason, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_evidence_engine(
    beats: list[Beat],
    morphology: Sequence[BeatMorphology],
    quality_after: QualityReport,
    rejected_regions: Sequence[tuple[float, float]] = (),
) -> list[MeasurementResult]:
    """Gate every measurement, each against its own evidence standard.

    Args:
        beats: detected beats with ``local_quality`` already attached.
        morphology: per-beat QRS boundary estimates.
        quality_after: post-recovery quality report, used for local noise.
        rejected_regions: spans whose recovery failed revalidation.

    Returns:
        One :class:`MeasurementResult` per measurement.
    """
    hf_at = [
        quality_after.feature_at(b.time, "hf_noise_ratio") for b in beats
    ]

    timing_verdicts = judge_beats(
        beats, BEAT_STANDARD_TIMING, rejected_regions=rejected_regions
    )
    morph_verdicts = judge_beats(
        beats,
        BEAT_STANDARD_MORPHOLOGY,
        rejected_regions=rejected_regions,
        morphology=morphology,
        hf_noise_at=hf_at,
    )

    return [
        gate_heart_rate(beats, timing_verdicts),
        gate_rr_interval(beats, timing_verdicts),
        gate_rr_consistency(beats, timing_verdicts),
        gate_qrs_duration(beats, morph_verdicts, morphology),
    ]
