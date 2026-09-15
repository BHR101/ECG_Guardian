"""Phase 8 -- physiological measurements.

This module answers "what is the number".  It does *not* decide whether the
number may be reported; that is :mod:`src.evidence`.  Keeping the two apart is
deliberate, because the whole point of the project is that computing a value
and being entitled to report it are different questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .peaks import Beat, physiologically_plausible

EPS = 1e-12

# QRS boundary search limits relative to the R peak (seconds).
QRS_SEARCH_PRE = 0.12
QRS_SEARCH_POST = 0.14
# Fraction of the peak QRS slope at which onset/offset is declared.
QRS_SLOPE_FRACTION = 0.15
# RR intervals within this fraction of the median count as "consistent".
RR_CONSISTENCY_TOLERANCE = 0.125


@dataclass
class BeatMorphology:
    """Per-beat QRS boundary estimate and the evidence for it."""

    beat_time: float
    onset: float = float("nan")      # seconds, absolute
    offset: float = float("nan")     # seconds, absolute
    duration_ms: float = float("nan")
    slope_peak: float = float("nan")
    slope_contrast: float = float("nan")  # QRS slope vs surrounding slope floor
    clear: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "beat_time": round(self.beat_time, 3),
            "onset": round(self.onset, 4) if not np.isnan(self.onset) else None,
            "offset": round(self.offset, 4) if not np.isnan(self.offset) else None,
            "duration_ms": round(self.duration_ms, 1)
            if not np.isnan(self.duration_ms)
            else None,
            "slope_contrast": round(self.slope_contrast, 2)
            if not np.isnan(self.slope_contrast)
            else None,
            "clear": self.clear,
            "reason": self.reason,
        }


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, width)
    if width == 1:
        return x
    return np.convolve(x, np.ones(width) / width, mode="same")


def _interp_crossing(
    d: np.ndarray, i_from: int, step: int, thr: float, hold: int
) -> float | None:
    """Walk from ``i_from`` in direction ``step`` to the first sustained crossing.

    Returns a fractional sample index (linearly interpolated) or ``None`` if the
    search leaves the window without finding one.
    """
    i = i_from
    n = d.size
    while 0 <= i < n:
        if d[i] < thr:
            # Require the low-slope state to persist, so a momentary dip in the
            # middle of the complex is not mistaken for its boundary.
            j_end = i + step * hold
            lo, hi = (min(i, j_end), max(i, j_end))
            lo, hi = max(0, lo), min(n, hi + 1)
            if np.all(d[lo:hi] < thr):
                prev = i - step
                if 0 <= prev < n and d[prev] != d[i]:
                    frac = (d[prev] - thr) / (d[prev] - d[i] + EPS)
                    return float(prev + step * np.clip(frac, 0.0, 1.0))
                return float(i)
        i += step
    return None


def qrs_boundaries(x: np.ndarray, fs: int, r_index: int) -> BeatMorphology:
    """Estimate QRS onset and offset for one beat from the slope envelope.

    The complex is bounded where the smoothed absolute slope falls below a
    fixed fraction of the peak QRS slope.  A boundary that is not found inside
    the search window is reported as *not clear* rather than guessed at.

    Args:
        x: processed waveform.
        fs: sampling rate.
        r_index: sample index of the R peak.

    Returns:
        A :class:`BeatMorphology`.  Never raises.
    """
    t_beat = float(r_index) / fs
    pre = int(QRS_SEARCH_PRE * fs)
    post = int(QRS_SEARCH_POST * fs)
    lo = r_index - pre
    hi = r_index + post + 1
    if lo < 0 or hi > x.size:
        return BeatMorphology(
            beat_time=t_beat, reason="beat too close to the record edge"
        )

    seg = x[lo:hi]
    d = _smooth(np.abs(np.gradient(seg)) * fs, max(3, int(round(0.012 * fs))))
    r_pos = pre

    if r_pos < 2 or r_pos > d.size - 3:
        return BeatMorphology(beat_time=t_beat, reason="search window too small")

    i_up = int(np.argmax(d[:r_pos + 1]))
    i_down = r_pos + int(np.argmax(d[r_pos:]))
    slope_peak = float(max(d[i_up], d[i_down]))
    if slope_peak <= EPS:
        return BeatMorphology(beat_time=t_beat, reason="no measurable QRS slope")

    # Slope floor: the quiet part of the window, away from the complex.
    edge = np.concatenate([d[: max(1, i_up - 2)], d[min(d.size, i_down + 3):]])
    slope_floor = float(np.median(edge)) if edge.size else 0.0
    slope_contrast = slope_peak / (slope_floor + EPS)

    thr = QRS_SLOPE_FRACTION * slope_peak
    hold = max(2, int(round(0.008 * fs)))

    on = _interp_crossing(d, i_up, -1, thr, hold)
    off = _interp_crossing(d, i_down, +1, thr, hold)

    if on is None or off is None:
        missing = "onset" if on is None else "offset"
        return BeatMorphology(
            beat_time=t_beat,
            slope_peak=slope_peak,
            slope_contrast=slope_contrast,
            reason=f"QRS {missing} not resolved within the search window",
        )

    onset_t = (lo + on) / fs
    offset_t = (lo + off) / fs
    duration_ms = (offset_t - onset_t) * 1000.0

    return BeatMorphology(
        beat_time=t_beat,
        onset=onset_t,
        offset=offset_t,
        duration_ms=duration_ms,
        slope_peak=slope_peak,
        slope_contrast=slope_contrast,
        clear=True,
        reason="onset and offset resolved",
    )


def measure_morphology(
    x: np.ndarray, fs: int, beats: list[Beat]
) -> list[BeatMorphology]:
    """Run :func:`qrs_boundaries` over a list of beats."""
    out: list[BeatMorphology] = []
    for b in beats:
        try:
            out.append(qrs_boundaries(x, fs, b.index))
        except Exception as exc:  # pragma: no cover - defensive
            out.append(
                BeatMorphology(beat_time=b.time, reason=f"morphology failed: {exc}")
            )
    return out


# ---------------------------------------------------------------------------
# Rhythm measurements
# ---------------------------------------------------------------------------
@dataclass
class RhythmMeasurements:
    """Raw rhythm quantities computed from a set of beats."""

    rr_intervals: np.ndarray = field(default_factory=lambda: np.asarray([]))
    rr_times: np.ndarray = field(default_factory=lambda: np.asarray([]))
    median_rr: float = float("nan")
    heart_rate: float = float("nan")
    consistency: float = float("nan")
    robust_cv: float = float("nan")
    n_intervals: int = 0
    n_excluded_intervals: int = 0


def rr_from_beats(beats: list[Beat], contiguous_only: bool = True) -> RhythmMeasurements:
    """Compute RR intervals and heart rate from an ordered list of beats.

    Args:
        beats: beats to use.  These should already be the *validated* ones.
        contiguous_only: when ``True``, only intervals between beats that were
            adjacent in the original detection are used, so a gap left by a
            rejected beat does not masquerade as a long RR interval.

    Returns:
        A :class:`RhythmMeasurements`.
    """
    m = RhythmMeasurements()
    if len(beats) < 2:
        return m

    intervals, times, excluded = [], [], 0
    for prev, cur in zip(beats[:-1], beats[1:]):
        rr = cur.time - prev.time
        if contiguous_only and not np.isclose(rr, prev.rr_after, atol=1e-6):
            # A beat between these two was dropped during validation.
            excluded += 1
            continue
        if not physiologically_plausible(rr):
            excluded += 1
            continue
        intervals.append(rr)
        times.append(cur.time)

    m.rr_intervals = np.asarray(intervals, dtype=float)
    m.rr_times = np.asarray(times, dtype=float)
    m.n_intervals = int(m.rr_intervals.size)
    m.n_excluded_intervals = excluded
    if m.n_intervals == 0:
        return m

    m.median_rr = float(np.median(m.rr_intervals))
    m.heart_rate = 60.0 / m.median_rr if m.median_rr > 0 else float("nan")
    m.consistency = rr_consistency(m.rr_intervals)
    m.robust_cv = robust_cv(m.rr_intervals)
    return m


def rr_consistency(rr: np.ndarray) -> float:
    """Fraction of RR intervals lying within +-12.5% of the median.

    A directly interpretable number: "how many of the measured beat-to-beat
    intervals agree with each other".
    """
    rr = np.asarray(rr, dtype=float)
    if rr.size == 0:
        return float("nan")
    med = float(np.median(rr))
    if med <= 0:
        return float("nan")
    return float(np.mean(np.abs(rr - med) / med <= RR_CONSISTENCY_TOLERANCE))


def robust_cv(rr: np.ndarray) -> float:
    """Interquartile range divided by the median -- an outlier-resistant CV."""
    rr = np.asarray(rr, dtype=float)
    if rr.size < 2:
        return float("nan")
    med = float(np.median(rr))
    if med <= 0:
        return float("nan")
    iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    return iqr / med
