"""Phase 5 + 6 -- artifact-specific recovery and mandatory revalidation.

Nothing in this module is novel signal processing: they are textbook
zero-phase Butterworth and notch filters.  What matters for this project is the
control flow around them.  A filter is never assumed to have worked.  Every
recovery attempt is followed by a fresh, independent quality assessment of the
result, and a region whose quality did not actually improve stays rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sps

from .config import (
    POWERLINE_FREQ,
    QUALITY_TRUSTED_MIN,
    RECOVERY_MIN_ABSOLUTE_GAIN,
    RECOVERY_MIN_POST_SCORE,
)
from .detection import ArtifactDetection
from .quality import QualityReport, assess_quality


# ---------------------------------------------------------------------------
# Primitive filters
# ---------------------------------------------------------------------------
def highpass(x: np.ndarray, fs: int, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass -- removes wandering baseline."""
    nyq = fs / 2.0
    wn = np.clip(cutoff / nyq, 1e-6, 0.99)
    sos = sps.butter(order, wn, btype="high", output="sos")
    return sps.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, fs: int, cutoff: float = 35.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass -- removes high-frequency noise."""
    nyq = fs / 2.0
    wn = np.clip(cutoff / nyq, 1e-6, 0.99)
    sos = sps.butter(order, wn, btype="low", output="sos")
    return sps.sosfiltfilt(sos, x)


def notch(
    x: np.ndarray, fs: int, freq: float = POWERLINE_FREQ, q: float = 30.0
) -> np.ndarray:
    """Zero-phase IIR notch at the mains frequency and its second harmonic."""
    nyq = fs / 2.0
    out = np.asarray(x, dtype=float)
    for harmonic in (1, 2):
        centre = freq * harmonic
        if centre >= 0.95 * nyq:
            continue
        b, a = sps.iirnotch(centre / nyq, q)
        out = sps.filtfilt(b, a, out)
    return out


# ---------------------------------------------------------------------------
# Recovery plan
# ---------------------------------------------------------------------------
@dataclass
class RegionOutcome:
    """Revalidation result for one detected artifact region."""

    start: float
    end: float
    artifact_type: str
    score_before: float
    score_after: float
    recovered: bool
    reason: str

    @property
    def gain(self) -> float:
        return self.score_after - self.score_before


@dataclass
class RecoveryReport:
    """Everything the dashboard needs to explain the recovery stage."""

    signal: np.ndarray
    methods: list[str] = field(default_factory=list)
    method_reasons: list[str] = field(default_factory=list)
    quality_before: QualityReport | None = None
    quality_after: QualityReport | None = None
    global_before: float = 0.0
    global_after: float = 0.0
    affected_before: float = float("nan")
    affected_after: float = float("nan")
    status: str = "NOT_REQUIRED"
    regions: list[RegionOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def method_label(self) -> str:
        return " + ".join(self.methods) if self.methods else "none (not required)"

    @property
    def rejected_regions(self) -> list[tuple[float, float]]:
        return [(r.start, r.end) for r in self.regions if not r.recovered]

    @property
    def recovered_regions(self) -> list[tuple[float, float]]:
        return [(r.start, r.end) for r in self.regions if r.recovered]

    def as_dict(self) -> dict[str, Any]:
        return {
            "methods": self.methods,
            "status": self.status,
            "global_before": round(self.global_before, 1),
            "global_after": round(self.global_after, 1),
            "affected_before": round(self.affected_before, 1)
            if not np.isnan(self.affected_before)
            else None,
            "affected_after": round(self.affected_after, 1)
            if not np.isnan(self.affected_after)
            else None,
            "regions": [
                {
                    "start": r.start,
                    "end": r.end,
                    "type": r.artifact_type,
                    "before": round(r.score_before, 1),
                    "after": round(r.score_after, 1),
                    "recovered": r.recovered,
                    "reason": r.reason,
                }
                for r in self.regions
            ],
        }


# Which filters each artifact class calls for, and whether filtering can even
# be expected to restore the underlying signal.
RECOVERY_PLAN: dict[str, dict[str, Any]] = {
    "baseline_wander": {
        "filters": ["baseline_correction"],
        "reason": "low-frequency drift is separable from the QRS band",
        "restorable": True,
    },
    "powerline": {
        "filters": ["mains_notch"],
        "reason": "narrowband interference is separable by a notch filter",
        "restorable": True,
    },
    "muscle_noise": {
        "filters": ["highfrequency_lowpass"],
        "reason": "noise energy sits mostly above the QRS band",
        "restorable": True,
    },
    "motion": {
        "filters": ["motion_aware_conservative"],
        "reason": (
            "motion energy overlaps the QRS band, so only a conservative "
            "band limitation is attempted and the result must be revalidated"
        ),
        "restorable": False,
    },
    "electrode_contact": {
        "filters": ["baseline_correction"],
        "reason": (
            "contact loss destroys the underlying signal; filtering cannot "
            "recreate absent QRS content"
        ),
        "restorable": False,
    },
    "unknown": {
        "filters": ["baseline_correction", "highfrequency_lowpass"],
        "reason": "conservative general-purpose band limitation",
        "restorable": False,
    },
}

# Structural revalidation.  A rising quality score is not sufficient evidence
# that recovery worked: removing a step discontinuity raises the score without
# restoring any of the QRS content it destroyed.  So for each artifact class we
# also re-measure the feature that *defined* that artifact and require it to
# have returned to a normal range.
#   (feature name, threshold, direction)  direction "below" or "above"
CLASS_DEFINING_FEATURE: dict[str, tuple[str, float, str]] = {
    "baseline_wander": ("baseline_drift_ratio", 0.10, "below"),
    "powerline": ("powerline_ratio", 0.03, "below"),
    "muscle_noise": ("hf_noise_ratio", 0.10, "below"),
    # Motion is defined by an amplitude excursion, so recovery has only worked
    # if the envelope is back at the record's normal scale.  QRS *contrast* is
    # deliberately not used here: a large motion spike raises the envelope peak
    # just as a QRS does, so contrast stays high while the beat is unusable.
    "motion": ("amplitude_ratio", 1.60, "below"),
    "electrode_contact": ("qrs_energy_ratio", 0.60, "above"),
    "unknown": ("amplitude_ratio", 1.60, "below"),
}

_FEATURE_LABELS = {
    "baseline_drift_ratio": "low-frequency drift",
    "powerline_ratio": "mains-band power fraction",
    "hf_noise_ratio": "high-frequency power fraction",
    "amplitude_ratio": "amplitude vs record reference",
    "qrs_energy_ratio": "QRS-band energy vs record",
}


# Share of the region's analysis windows that must individually clear the
# structural criterion.  A regional *mean* can be carried over the line by one
# unrepresentative window -- a step discontinuity inside a dropout, say, which
# contributes plenty of QRS-band energy while restoring no actual signal -- so
# the requirement is that the defect has cleared across the region, not on
# average across it.
STRUCTURAL_WINDOW_FRACTION = 0.70


def _structural_check(
    det_type: str, quality_after: QualityReport, start: float, end: float
) -> tuple[bool, str]:
    """Re-measure the artifact's defining feature after recovery.

    Returns (passed, explanation).  An unmeasurable feature fails closed.
    """
    name, threshold, direction = CLASS_DEFINING_FEATURE.get(
        det_type, CLASS_DEFINING_FEATURE["unknown"]
    )
    label = _FEATURE_LABELS.get(name, name)

    values = [
        w.features[name]
        for w in quality_after.windows
        if w.end > start and w.start < end and name in w.features
    ]
    if not values:
        return False, f"{label} could not be re-measured after recovery"

    arr = np.asarray(values, dtype=float)
    passing = arr < threshold if direction == "below" else arr > threshold
    share = float(np.mean(passing))
    ok = share >= STRUCTURAL_WINDOW_FRACTION

    typical = float(np.median(arr))
    sense = "<" if direction == "below" else ">"
    fmt = "{:.3f}" if direction == "below" else "{:.2f}"
    cmp_txt = (
        f"{share:.0%} of windows meet {sense} {threshold:.2f}, "
        f"typical {fmt.format(typical)}"
    )
    verb = "normalised" if ok else "did not normalise"
    return ok, f"{label} {verb} across the region ({cmp_txt})"


_FILTER_LABELS = {
    "baseline_correction": "baseline correction (0.5 Hz high-pass)",
    "mains_notch": f"{POWERLINE_FREQ:.0f} Hz notch filter",
    "highfrequency_lowpass": "high-frequency low-pass (35 Hz)",
    "motion_aware_conservative": "motion-aware conservative band limitation (1-30 Hz)",
}


def _apply_filter(name: str, x: np.ndarray, fs: int) -> np.ndarray:
    if name == "baseline_correction":
        return highpass(x, fs, cutoff=0.5)
    if name == "mains_notch":
        return notch(x, fs, POWERLINE_FREQ)
    if name == "highfrequency_lowpass":
        return lowpass(x, fs, cutoff=35.0)
    if name == "motion_aware_conservative":
        return lowpass(highpass(x, fs, cutoff=1.0), fs, cutoff=30.0)
    raise ValueError(f"unknown recovery filter {name!r}")


def recover(
    x: np.ndarray,
    fs: int,
    detections: list[ArtifactDetection],
    quality_before: QualityReport,
) -> RecoveryReport:
    """Attempt artifact-specific recovery, then revalidate it.

    Filters are chosen from the *detected* artifact classes, applied over the
    whole record (zero-phase filters have no edge-selective advantage here),
    and the outcome is then judged per detected region.

    Args:
        x: the raw waveform.
        fs: sampling rate.
        detections: output of :func:`src.detection.detect_artifacts`.
        quality_before: the quality report for ``x``.

    Returns:
        A :class:`RecoveryReport`.  This function does not raise on filter
        failure; if a filter errors the signal is passed through unchanged and
        a note is recorded, so the pipeline always produces a result.
    """
    x = np.asarray(x, dtype=float)
    report = RecoveryReport(signal=x.copy(), quality_before=quality_before)
    report.global_before = quality_before.score

    if not detections:
        report.status = "NOT_REQUIRED"
        report.quality_after = quality_before
        report.global_after = quality_before.score
        report.notes.append("No disturbance detected; signal passed through unchanged.")
        return report

    # -- choose the filter chain -------------------------------------------
    chain: list[str] = []
    reasons: list[str] = []
    for det in detections:
        plan = RECOVERY_PLAN.get(det.type, RECOVERY_PLAN["unknown"])
        for f in plan["filters"]:
            if f not in chain:
                chain.append(f)
                reasons.append(f"{det.label}: {plan['reason']}")

    # motion-aware band limitation subsumes plain baseline correction
    if "motion_aware_conservative" in chain and "baseline_correction" in chain:
        chain.remove("baseline_correction")

    y = x.copy()
    applied: list[str] = []
    for name in chain:
        try:
            y = _apply_filter(name, y, fs)
            applied.append(name)
        except Exception as exc:  # pragma: no cover - defensive
            report.notes.append(f"Filter {name} failed ({exc}); skipped.")

    report.signal = y
    report.methods = [_FILTER_LABELS.get(n, n) for n in applied]
    report.method_reasons = reasons

    # -- REVALIDATION: score the result independently -----------------------
    # The processed signal is assessed on its own terms (its own amplitude
    # reference).  Reusing the raw reference would make a successful baseline
    # correction look like an amplitude collapse.
    try:
        quality_after = assess_quality(y, fs)
    except Exception as exc:  # pragma: no cover - defensive
        report.notes.append(f"Revalidation failed ({exc}); recovery not trusted.")
        report.status = "FAILED"
        report.quality_after = quality_before
        report.global_after = quality_before.score
        for det in detections:
            report.regions.append(
                RegionOutcome(
                    det.start, det.end, det.type,
                    det.mean_score, det.mean_score, False,
                    "revalidation could not be completed",
                )
            )
        return report

    report.quality_after = quality_after
    report.global_after = quality_after.score

    aff_before, aff_after = [], []
    for det in detections:
        before = quality_before.mean_score_between(det.start, det.end)
        after = quality_after.mean_score_between(det.start, det.end)
        aff_before.append(before)
        aff_after.append(after)

        gain = after - before

        # Test 1 -- is the region acceptable now?
        # Two ways to satisfy this.  Either the region already clears the
        # trusted threshold -- in which case demanding a further improvement is
        # incoherent, since there was nothing left to gain -- or it improved by
        # a real margin and cleared the post-recovery floor.  Without the first
        # branch a mildly degraded stretch sitting at 70/100 is condemned
        # because filtering could only take it to 71, and every measurement
        # that depended on it is refused for no reason.
        if after >= QUALITY_TRUSTED_MIN:
            score_ok = True
            score_why = (
                f"quality {before:.0f} -> {after:.0f}, at or above the trusted "
                f"threshold ({QUALITY_TRUSTED_MIN:.0f})"
            )
        elif gain < RECOVERY_MIN_ABSOLUTE_GAIN:
            score_ok = False
            score_why = (
                f"quality moved only {before:.0f} -> {after:.0f} "
                f"(needed +{RECOVERY_MIN_ABSOLUTE_GAIN:.0f})"
            )
        elif after < RECOVERY_MIN_POST_SCORE:
            score_ok = False
            score_why = (
                f"quality rose {before:.0f} -> {after:.0f} but stayed below the "
                f"minimum post-recovery score ({RECOVERY_MIN_POST_SCORE:.0f})"
            )
        else:
            score_ok = True
            score_why = (
                f"quality rose {before:.0f} -> {after:.0f}, clearing the minimum "
                f"post-recovery score ({RECOVERY_MIN_POST_SCORE:.0f})"
            )

        # Test 2 -- did the property that defined the artifact actually go away?
        struct_ok, struct_why = _structural_check(
            det.type, quality_after, det.start, det.end
        )

        ok = score_ok and struct_ok
        why = f"{score_why}; {struct_why}"

        report.regions.append(
            RegionOutcome(det.start, det.end, det.type, before, after, ok, why)
        )

    report.affected_before = float(np.mean(aff_before)) if aff_before else float("nan")
    report.affected_after = float(np.mean(aff_after)) if aff_after else float("nan")

    n_ok = sum(1 for r in report.regions if r.recovered)
    if n_ok == len(report.regions):
        report.status = "VALIDATED"
    elif n_ok == 0:
        report.status = "FAILED"
    else:
        report.status = "PARTIAL"

    return report
