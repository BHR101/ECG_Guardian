"""Phase 7 -- R-peak detection with per-beat evidence.

A conventional Pan-Tompkins-style chain (band-pass, differentiate, square,
integrate, adaptive threshold).  The algorithm is not the contribution; what
matters downstream is that every detected beat carries measurable evidence
about how trustworthy it is.

This module never sees the ground-truth peak positions.  It takes a waveform
and nothing else.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import signal as sps

from .config import (
    INTEGRATION_WINDOW_SEC,
    QRS_BAND,
    REFRACTORY_SEC,
    RR_MAX_SEC,
    RR_MIN_SEC,
)

EPS = 1e-12

# Fraction of the locally expected peak height a candidate must reach.
ADAPTIVE_THRESHOLD_FRACTION = 0.35
# Half-width, in seconds, of the neighbourhood used for the local threshold.
LOCAL_REFERENCE_SEC = 4.0
# Half-width of the beat window used for template matching.
BEAT_HALF_WIDTH_SEC = 0.18


@dataclass
class Beat:
    """One detected beat and the evidence supporting it."""

    index: int                  # sample index of the R peak
    time: float                 # seconds
    amplitude: float            # processed-signal amplitude at the peak
    prominence: float           # topographic prominence
    relative_amplitude: float   # amplitude / median beat amplitude
    relative_prominence: float  # prominence / median beat prominence
    template_correlation: float  # correlation with the median beat shape
    local_quality: float = float("nan")   # filled in by the pipeline
    rr_before: float = float("nan")
    rr_after: float = float("nan")
    valid: bool = True
    reject_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "amplitude": round(self.amplitude, 3),
            "relative_amplitude": round(self.relative_amplitude, 3),
            "relative_prominence": round(self.relative_prominence, 3),
            "template_correlation": round(self.template_correlation, 3),
            "local_quality": round(self.local_quality, 1)
            if not np.isnan(self.local_quality)
            else None,
            "valid": self.valid,
            "reject_reasons": list(self.reject_reasons),
        }


@dataclass
class PeakReport:
    """All detected beats plus the intermediate signals used to find them."""

    beats: list[Beat]
    detection_signal: np.ndarray
    sampling_rate: int

    @property
    def indices(self) -> np.ndarray:
        return np.asarray([b.index for b in self.beats], dtype=int)

    @property
    def times(self) -> np.ndarray:
        return np.asarray([b.time for b in self.beats], dtype=float)

    @property
    def rr_intervals(self) -> np.ndarray:
        """All RR intervals in seconds, in order."""
        t = self.times
        return np.diff(t) if t.size > 1 else np.asarray([], dtype=float)


def _qrs_enhance(x: np.ndarray, fs: int) -> np.ndarray:
    """Band-pass, differentiate, square and integrate -- the detection signal."""
    nyq = fs / 2.0
    lo, hi = QRS_BAND
    hi = min(hi, 0.95 * nyq)
    sos = sps.butter(3, [lo / nyq, hi / nyq], btype="band", output="sos")
    band = sps.sosfiltfilt(sos, x)
    deriv = np.gradient(band) * fs
    squared = deriv ** 2
    win = max(3, int(INTEGRATION_WINDOW_SEC * fs))
    kernel = np.ones(win) / win
    return np.convolve(squared, kernel, mode="same")


def _local_threshold(
    positions: np.ndarray, heights: np.ndarray, fs: int
) -> np.ndarray:
    """Adaptive per-candidate threshold from nearby candidate heights.

    Using a *local* median keeps the detector working when beat amplitude
    drifts, while an absolute floor derived from the whole record stops it
    inventing beats inside a flat or dropped-out stretch.
    """
    if positions.size == 0:
        return np.asarray([], dtype=float)
    half = LOCAL_REFERENCE_SEC * fs
    global_ref = float(np.median(heights))
    thresholds = np.empty(positions.size, dtype=float)
    for i, p in enumerate(positions):
        near = heights[np.abs(positions - p) <= half]
        local_ref = float(np.median(near)) if near.size else global_ref
        # Blend local and global so one very loud region cannot raise the bar
        # for the whole record, nor a quiet one lower it to noise level.
        ref = max(0.5 * local_ref + 0.5 * global_ref, 0.30 * global_ref)
        thresholds[i] = ADAPTIVE_THRESHOLD_FRACTION * ref
    return thresholds


def _refine_to_peak(x: np.ndarray, idx: int, fs: int, half_sec: float = 0.06) -> int:
    """Snap a detection to the local maximum of the processed signal.

    The detection chain is zero-phase and the integrator is symmetric, so the
    candidate is already close; this only removes a few samples of smearing.
    Assumes upright R waves, which holds for the synthetic generator.
    """
    half = max(1, int(half_sec * fs))
    lo = max(0, idx - half)
    hi = min(x.size, idx + half + 1)
    if hi <= lo:
        return idx
    return lo + int(np.argmax(x[lo:hi]))


def _beat_templates(
    x: np.ndarray, indices: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-beat windows and the median beat template."""
    half = int(BEAT_HALF_WIDTH_SEC * fs)
    windows = []
    for idx in indices:
        lo, hi = idx - half, idx + half + 1
        if lo < 0 or hi > x.size:
            # Pad edge beats so every beat gets a comparable window.
            seg = np.zeros(2 * half + 1)
            a, b = max(0, lo), min(x.size, hi)
            seg[a - lo:(a - lo) + (b - a)] = x[a:b]
        else:
            seg = x[lo:hi]
        windows.append(seg)
    arr = np.asarray(windows, dtype=float) if windows else np.zeros((0, 2 * half + 1))
    template = np.median(arr, axis=0) if arr.size else np.zeros(2 * half + 1)
    return arr, template


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def detect_r_peaks(x: np.ndarray, fs: int) -> PeakReport:
    """Detect R peaks in a processed ECG and describe each beat.

    Args:
        x: the (recovered / processed) waveform.  Ground truth is never used.
        fs: sampling rate in Hz.

    Returns:
        A :class:`PeakReport`.  Returns an empty beat list rather than raising
        when the signal contains nothing peak-like.
    """
    x = np.asarray(x, dtype=float)
    integ = _qrs_enhance(x, fs)

    refr = max(1, int(REFRACTORY_SEC * fs))
    candidates, _ = sps.find_peaks(integ, distance=refr)
    if candidates.size == 0:
        return PeakReport(beats=[], detection_signal=integ, sampling_rate=fs)

    heights = integ[candidates]
    thresholds = _local_threshold(candidates, heights, fs)
    kept = candidates[heights >= thresholds]
    if kept.size == 0:
        return PeakReport(beats=[], detection_signal=integ, sampling_rate=fs)

    refined = np.asarray([_refine_to_peak(x, int(i), fs) for i in kept], dtype=int)
    refined = np.unique(refined)

    # Enforce the refractory period again after refinement.
    final: list[int] = []
    for idx in refined:
        if final and idx - final[-1] < refr:
            # Keep whichever of the two has the larger detection energy.
            if integ[idx] > integ[final[-1]]:
                final[-1] = int(idx)
            continue
        final.append(int(idx))
    indices = np.asarray(final, dtype=int)

    # A prominence of zero is a meaningful answer here, not a problem: it says
    # the candidate does not stand out at all, and the evidence engine will
    # refuse it.  SciPy warns about it, so silence just that warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*prominence of 0.*")
        prominences = (
            sps.peak_prominences(x, indices)[0]
            if indices.size
            else np.asarray([], dtype=float)
        )
    amplitudes = x[indices]
    med_amp = float(np.median(np.abs(amplitudes))) + EPS
    med_prom = float(np.median(prominences)) + EPS

    windows, template = _beat_templates(x, indices, fs)
    correlations = np.asarray(
        [_correlate(w, template) for w in windows], dtype=float
    )

    beats: list[Beat] = []
    for i, idx in enumerate(indices):
        beats.append(
            Beat(
                index=int(idx),
                time=float(idx) / fs,
                amplitude=float(amplitudes[i]),
                prominence=float(prominences[i]),
                relative_amplitude=float(abs(amplitudes[i]) / med_amp),
                relative_prominence=float(prominences[i] / med_prom),
                template_correlation=float(correlations[i]),
            )
        )

    # Attach neighbouring RR intervals to each beat.
    for i, beat in enumerate(beats):
        if i > 0:
            beat.rr_before = beat.time - beats[i - 1].time
        if i < len(beats) - 1:
            beat.rr_after = beats[i + 1].time - beat.time

    return PeakReport(beats=beats, detection_signal=integ, sampling_rate=fs)


def rescore_beats(
    x: np.ndarray,
    fs: int,
    beats: list[Beat],
    reference_mask: Sequence[bool],
    min_reference_beats: int = 3,
) -> None:
    """Re-normalise each beat's relative metrics against a chosen subset.

    On first pass every beat is compared with the median of *all* detected
    beats.  That is the right default, but it breaks down when most of the
    record is corrupted: if two-thirds of the "beats" are dropout residuals,
    the median describes the residuals, and the genuinely good beats then
    measure as wild outliers and get rejected.

    Once the pipeline knows which beats sit in trustworthy parts of the record,
    it calls this to rebuild the amplitude, prominence and shape references
    from those beats only -- so each beat is judged against what a *good* beat
    in this record actually looks like.  Beats are modified in place.

    Args:
        x: the processed waveform the beats were detected in.
        fs: sampling rate.
        beats: the detected beats.
        reference_mask: one flag per beat; ``True`` marks a beat eligible to
            define the reference.
        min_reference_beats: below this many eligible beats the existing
            references are kept, since too small a subset is not a reference.
    """
    if not beats:
        return
    mask = np.asarray(list(reference_mask), dtype=bool)
    if mask.size != len(beats) or int(mask.sum()) < min_reference_beats:
        return

    indices = np.asarray([b.index for b in beats], dtype=int)
    amplitudes = np.asarray([b.amplitude for b in beats], dtype=float)
    prominences = np.asarray([b.prominence for b in beats], dtype=float)

    med_amp = float(np.median(np.abs(amplitudes[mask]))) + EPS
    med_prom = float(np.median(prominences[mask])) + EPS

    windows, _ = _beat_templates(x, indices, fs)
    template = (
        np.median(windows[mask], axis=0) if windows.size else np.zeros(1)
    )

    for i, beat in enumerate(beats):
        beat.relative_amplitude = float(abs(amplitudes[i]) / med_amp)
        beat.relative_prominence = float(prominences[i] / med_prom)
        if windows.size:
            beat.template_correlation = _correlate(windows[i], template)


def physiologically_plausible(rr: float) -> bool:
    """Sanity rail on an RR interval.  Not a diagnosis, just a range check."""
    return bool(RR_MIN_SEC <= rr <= RR_MAX_SEC)
