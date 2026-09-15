"""Phase 4 -- interpretable artifact detection, localisation and classification.

There is no trained model here.  Detection is a rule over the same measured
features that drive the quality score, which means every decision can be shown
to a reviewer as an arithmetic statement about the waveform.

Localisation works at the resolution of the quality analysis windows.  A region
is reported over the span of the *centres* of the disturbed windows, widened by
half a hop, because the centre is where the evidence for that window actually
sits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .artifacts import HUMAN_LABELS
from .config import POWERLINE_FREQ, QUALITY_HOP_SEC, QUALITY_TRUSTED_MIN
from .quality import QualityReport, WindowQuality

EPS = 1e-12

# How much penalty a region must carry before we are willing to name a type.
MIN_CLASS_EVIDENCE = 6.0

# Contiguous disturbed regions closer together than this are merged.
MERGE_GAP_SEC = 0.75

# A disturbance must be visible in at least two overlapping analysis windows
# before it is reported.  With a 1.0 s window and a 0.5 s hop, one window spans
# 0.5 s of centre-time and two span 1.0 s, so this threshold is the statement
# "one window's worth of evidence is not enough".
MIN_REGION_SEC = 0.9


@dataclass
class ArtifactDetection:
    """One localised, classified disturbance."""

    type: str
    start: float
    end: float
    confidence: float
    evidence: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    class_scores: dict[str, float] = field(default_factory=dict)
    mean_score: float = 0.0
    min_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def label(self) -> str:
        return HUMAN_LABELS.get(self.type, self.type)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        d["duration"] = self.duration
        return d


def _class_scores(pens: dict[str, float], feats: dict[str, float]) -> dict[str, float]:
    """Map the quality penalty breakdown onto artifact-class evidence scores.

    Each class is defined by the *combination* of penalties it produces:

    * baseline wander  -- low-frequency drift and nothing else
    * power-line       -- narrowband energy at the mains frequency
    * muscle noise     -- broadband high-frequency energy
    * motion           -- amplitude excursion that also obscures the QRS,
                          usually with some drift
    * electrode contact-- QRS energy missing relative to the rest of the record
    """
    amp_ratio = feats.get("amplitude_ratio", 1.0)
    amp_pen = pens.get("amplitude_anomaly", 0.0)
    # Split the amplitude penalty by direction: too big points at motion,
    # too small points at a failing electrode.
    amp_excess = amp_pen if amp_ratio >= 1.0 else 0.0
    amp_deficit = amp_pen if amp_ratio < 1.0 else 0.0

    # An amplitude excursion on its own is not evidence of motion: it is only
    # motion if the complexes are also being obscured.  Without this coupling,
    # a record that is mostly dropout deflates the amplitude reference so far
    # that its few *intact* stretches look like large excursions and get
    # labelled motion.
    vis_pen = pens.get("qrs_visibility", 0.0)
    contrast_lost = float(np.clip(vis_pen / 15.0, 0.0, 1.0))

    # A baseline that moves while the signal stays intact is wander.  The same
    # baseline shift accompanied by missing QRS energy is a failing electrode,
    # which is the more specific finding -- so drop the wander evidence when
    # the QRS content has gone.
    drop_pen = pens.get("dropout", 0.0)
    signal_intact = 1.0 - float(np.clip(drop_pen / 30.0, 0.0, 1.0))

    return {
        "baseline_wander": pens.get("baseline_drift", 0.0) * signal_intact,
        "powerline": pens.get("powerline_interference", 0.0),
        "muscle_noise": pens.get("high_frequency_noise", 0.0),
        "motion": amp_excess * contrast_lost + vis_pen,
        "electrode_contact": pens.get("dropout", 0.0) + amp_deficit,
    }


def _evidence_strings(kind: str, feats: dict[str, float], pens: dict[str, float]) -> list[str]:
    """Human-readable statements of the measurements that drove the decision."""
    ev: list[str] = []
    if kind == "baseline_wander":
        ev.append(
            f"sub-0.7 Hz drift is {feats['baseline_drift_ratio'] * 100:.0f}% of "
            f"reference QRS amplitude"
        )
        ev.append("high-frequency and mains bands are clean")
    elif kind == "powerline":
        ev.append(
            f"{POWERLINE_FREQ:.0f} Hz band holds "
            f"{feats['powerline_ratio'] * 100:.0f}% of in-band power"
        )
        ev.append("energy is narrowband, not broadband")
    elif kind == "muscle_noise":
        ev.append(
            f"40-100 Hz band holds {feats['hf_noise_ratio'] * 100:.0f}% of in-band power"
        )
        ev.append("broadband high-frequency energy, QRS still visible")
    elif kind == "motion":
        ev.append(
            f"amplitude is {feats['amplitude_ratio']:.2f}x the record reference"
        )
        ev.append(
            f"QRS contrast fell to {feats['qrs_visibility']:.1f} "
            f"(clean reference is above 7)"
        )
        if pens.get("baseline_drift", 0.0) > 3.0:
            ev.append("accompanied by low-frequency excursion")
    elif kind == "electrode_contact":
        ev.append(
            f"QRS-band energy is {feats['qrs_energy_ratio'] * 100:.0f}% of the "
            f"record-wide level"
        )
        ev.append(
            f"amplitude is {feats['amplitude_ratio']:.2f}x the record reference"
        )
    else:
        ev.append("quality degraded but no single feature dominates")
        top = sorted(pens.items(), key=lambda kv: -kv[1])[:2]
        ev.extend(f"{k.replace('_', ' ')} penalty {v:.0f}" for k, v in top if v > 1.0)
    return ev


def _window_class(w: WindowQuality) -> str:
    """The artifact class best supported by one window's own measurements."""
    scores = _class_scores(w.penalties, w.features)
    kind, value = max(scores.items(), key=lambda kv: kv[1])
    return kind if value >= MIN_CLASS_EVIDENCE else "unknown"


def _group_windows(
    windows: list[WindowQuality], threshold: float, hop: float
) -> list[list[WindowQuality]]:
    """Group disturbed windows into regions of a single artifact class.

    Windows are classified individually first, then grouped only while they are
    both adjacent *and* of the same class.  Averaging features across a region
    that contains two different disturbances would otherwise produce a blend
    that matches neither.
    """
    flagged = [(w, _window_class(w)) for w in windows if w.score < threshold]
    if not flagged:
        return []

    # Smooth the label sequence: a single window disagreeing with both of its
    # neighbours is noise in the classifier, not a one-second artifact of a
    # different kind.
    labels = [k for _, k in flagged]
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
            smoothed[i] = labels[i - 1]
    flagged = [(w, k) for (w, _), k in zip(flagged, smoothed)]

    groups: list[list[WindowQuality]] = [[flagged[0][0]]]
    kinds: list[str] = [flagged[0][1]]
    for w, kind in flagged[1:]:
        prev = groups[-1][-1]
        centre_gap = (0.5 * (w.start + w.end)) - (0.5 * (prev.start + prev.end))
        if centre_gap <= MERGE_GAP_SEC + hop * 0.5 and kind == kinds[-1]:
            groups[-1].append(w)
        else:
            groups.append([w])
            kinds.append(kind)
    return groups


def detect_artifacts(
    quality: QualityReport,
    threshold: float = QUALITY_TRUSTED_MIN,
    hop: float = QUALITY_HOP_SEC,
    duration: float | None = None,
) -> list[ArtifactDetection]:
    """Find, localise and classify disturbances from a quality report.

    Args:
        quality: the windowed quality report for the signal under test.
        threshold: window score below which a window counts as disturbed.
        hop: the hop used when building the quality report.
        duration: record duration, used to clamp region bounds.

    Returns:
        A list of :class:`ArtifactDetection`, ordered in time.  Never raises on
        unusual input -- an unclassifiable region is reported as ``"unknown"``.
    """
    detections: list[ArtifactDetection] = []
    if not quality.windows:
        return detections

    limit = duration if duration is not None else quality.windows[-1].end

    for group in _group_windows(quality.windows, threshold, hop):
        centres = [0.5 * (w.start + w.end) for w in group]
        start = max(0.0, centres[0] - 0.5 * hop)
        end = min(limit, centres[-1] + 0.5 * hop)
        if end - start < MIN_REGION_SEC:
            continue

        # Aggregate with the median, not the mean: one unrepresentative window
        # inside a region (a step discontinuity in the middle of a dropout, for
        # instance) should not redefine what the whole region looks like.
        feats = {
            k: float(np.median([w.features[k] for w in group]))
            for k in group[0].features
        }
        pens = {
            k: float(np.median([w.penalties[k] for w in group]))
            for k in group[0].penalties
        }

        scores = _class_scores(pens, feats)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        winner, win_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if win_score < MIN_CLASS_EVIDENCE:
            kind = "unknown"
            confidence = 0.35
        else:
            kind = winner
            total = sum(v for v in scores.values() if v > 0.0) + EPS
            dominance = win_score / total
            strength = float(np.clip(win_score / 50.0, 0.0, 1.0))
            separation = float(np.clip((win_score - runner_up) / (win_score + EPS), 0.0, 1.0))
            confidence = float(
                np.clip(0.40 * dominance + 0.35 * strength + 0.25 * separation, 0.0, 0.98)
            )

        detections.append(
            ArtifactDetection(
                type=kind,
                start=round(start, 2),
                end=round(end, 2),
                confidence=round(confidence, 3),
                evidence=_evidence_strings(kind, feats, pens),
                features={k: round(v, 4) for k, v in feats.items()},
                class_scores={k: round(v, 1) for k, v in scores.items()},
                mean_score=round(float(np.mean([w.score for w in group])), 1),
                min_score=round(float(np.min([w.score for w in group])), 1),
            )
        )

    return detections


def merge_detection_regions(
    detections: list[ArtifactDetection],
) -> list[tuple[float, float]]:
    """Flatten detections into a list of non-overlapping (start, end) spans."""
    if not detections:
        return []
    spans = sorted((d.start, d.end) for d in detections)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def in_any_region(t: float, regions: list[tuple[float, float]]) -> bool:
    return any(s <= t < e for s, e in regions)
