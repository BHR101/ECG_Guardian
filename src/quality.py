"""Phase 3 -- transparent windowed signal-quality assessment.

Every number produced here is a measured property of the waveform.  There are
no learned weights and no random confidence values: the score is a documented
penalty function over measurable features, so the dashboard can always say
*why* it changed.

The output is deliberately named the "Prototype Signal Quality Index".  It is
not a clinically validated measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sps

from .config import (
    POWERLINE_FREQ,
    QUALITY_DEGRADED_MIN,
    QUALITY_HOP_SEC,
    QUALITY_TRUSTED_MIN,
    QUALITY_WINDOW_SEC,
)

EPS = 1e-12

# ---------------------------------------------------------------------------
# Penalty calibration.  Each entry is (knee, saturation, max_penalty):
# below `knee` the feature costs nothing, at/above `saturation` it costs
# `max_penalty`, and it interpolates linearly in between.
# These constants were tuned against measured feature values for the synthetic
# clean signal and each injected artifact type (see tools/calibrate.py).
# ---------------------------------------------------------------------------
PENALTIES: dict[str, tuple[float, float, float]] = {
    # feature name          knee    saturation   max penalty
    "baseline_drift_ratio": (0.06, 0.35, 55.0),
    "hf_noise_ratio": (0.05, 0.45, 45.0),
    "powerline_ratio": (0.01, 0.20, 40.0),
    "amplitude_excess": (1.70, 4.00, 50.0),
    "amplitude_deficit": (0.55, 0.15, 60.0),   # note: inverted direction
    "dropout_index": (0.25, 0.80, 60.0),
    "qrs_visibility_deficit": (0.0, 1.0, 50.0),
}

QRS_VISIBILITY_GOOD = 7.0   # envelope peak / envelope median at which QRS is clear
QRS_VISIBILITY_BAD = 2.2    # at/below this the QRS is not distinguishable

# QRS *contrast* cannot be judged inside a window shorter than one beat
# interval: at 48 BPM the beats are 1.25 s apart, so a 1.0 s window can
# legitimately fall between two of them and contain no complex at all.  That
# one feature is therefore measured over a wider context centred on the window.
ENVELOPE_CONTEXT_SEC = 1.5

# Dropout is a question about *time*, not about a single window: has this
# record gone unusually long without producing any QRS-band energy?  A window
# is scored by the length of the QRS-free gap it falls inside, relative to how
# often this record normally produces a complex.  Padding the window instead
# would let a neighbouring beat leak in and hide a genuine signal loss.
QRS_EVENT_FRACTION = 0.5     # of the record reference envelope

# A complex is a *transient*: the trace leaves the baseline and comes back to
# it.  A contact-loss step is a *level change*: the trace leaves and stays.
# Both deposit broadband energy, so an energy-only measure cannot tell them
# apart -- which is exactly how a step hides a dropout, by splitting the
# QRS-free gap in two and making the missing signal look occupied.
# Measured separation on the synthetic corpus: real beats sit at 0.016-0.037
# typically and never exceed 0.29 even under severe baseline wander or at
# 180 BPM, while contact-loss steps measure ~0.80.
STEP_REJECT_RATIO = 0.45
QRS_EVENT_MERGE_SEC = 0.2    # excursions closer than this are one complex
DROPOUT_GAP_FLOOR_SEC = 1.6  # below ~37 BPM, so a slow rhythm is never a dropout
DROPOUT_GAP_MULTIPLE = 2.2   # ... or this many typical beat intervals

# The energy view of dropout: QRS-band energy over a beat-spanning context,
# as a fraction of the record reference.  At/above FULL there is no dropout;
# at/below NONE the signal is gone.
ENERGY_DROPOUT_FULL = 0.60
ENERGY_DROPOUT_NONE = 0.10


def _linear_penalty(value: float, knee: float, sat: float, max_pen: float) -> float:
    """Piecewise-linear penalty; handles increasing and decreasing features."""
    if sat == knee:
        return 0.0
    frac = (value - knee) / (sat - knee)
    return float(np.clip(frac, 0.0, 1.0) * max_pen)


@dataclass
class WindowQuality:
    """Quality of one analysis window."""

    index: int
    start: float
    end: float
    score: float
    features: dict[str, float]
    penalties: dict[str, float]

    @property
    def label(self) -> str:
        if self.score >= QUALITY_TRUSTED_MIN:
            return "TRUSTED"
        if self.score >= QUALITY_DEGRADED_MIN:
            return "DEGRADED"
        return "REJECTED"


@dataclass
class QualityReport:
    """Whole-record quality summary plus the per-window detail behind it."""

    score: float
    windows: list[WindowQuality]
    evidence: dict[str, Any] = field(default_factory=dict)
    reference_p2p: float = 1.0

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([w.score for w in self.windows], dtype=float)

    @property
    def centers(self) -> np.ndarray:
        return np.asarray(
            [0.5 * (w.start + w.end) for w in self.windows], dtype=float
        )

    def score_at(self, t: float) -> float:
        """Quality score of the window nearest to time ``t``."""
        if not self.windows:
            return float("nan")
        idx = int(np.argmin(np.abs(self.centers - t)))
        return self.windows[idx].score

    def mean_score_between(self, start: float, end: float) -> float:
        """Mean window score over windows overlapping ``[start, end)``."""
        vals = [w.score for w in self.windows if w.end > start and w.start < end]
        return float(np.mean(vals)) if vals else float("nan")

    def mean_feature_between(self, start: float, end: float, name: str) -> float:
        """Mean value of one feature over windows overlapping ``[start, end)``."""
        vals = [
            w.features[name]
            for w in self.windows
            if w.end > start and w.start < end and name in w.features
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def feature_at(self, t: float, name: str) -> float:
        if not self.windows:
            return float("nan")
        idx = int(np.argmin(np.abs(self.centers - t)))
        return self.windows[idx].features.get(name, float("nan"))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _bandpass_envelope(x: np.ndarray, fs: int) -> np.ndarray:
    """Smoothed magnitude envelope of the 5-18 Hz (QRS) band."""
    nyq = fs / 2.0
    high = min(18.0, 0.95 * nyq)
    low = min(5.0, 0.5 * high)
    sos = sps.butter(3, [low / nyq, high / nyq], btype="band", output="sos")
    band = sps.sosfiltfilt(sos, x)
    env = np.abs(band)
    win = max(3, int(0.05 * fs))
    kernel = np.ones(win) / win
    return np.convolve(env, kernel, mode="same")


def _lowfreq_component(x: np.ndarray, fs: int) -> np.ndarray:
    """The sub-0.7 Hz content, i.e. the wandering baseline itself."""
    nyq = fs / 2.0
    sos = sps.butter(2, 0.7 / nyq, btype="low", output="sos")
    return sps.sosfiltfilt(sos, x)


def baseline_return_ratio(
    x: np.ndarray, fs: int, t: float, reference_p2p: float
) -> float:
    """How far the trace fails to return to where it started, across an event.

    Compares the median level just before an event with the median level just
    after it, both taken over 120 ms windows that straddle the complex without
    touching its peak.  A QRS leaves both windows sitting on the isoelectric
    line, so the difference is near zero.  A step discontinuity moves the
    second window by the full height of the step.

    Short windows are used deliberately: a wandering baseline moves slowly and
    barely changes across 320 ms, so drift does not masquerade as a step.
    """
    i = int(round(t * fs))
    a0, a1 = max(0, i - int(0.16 * fs)), max(1, i - int(0.04 * fs))
    b0, b1 = min(x.size - 1, i + int(0.04 * fs)), min(x.size, i + int(0.16 * fs))
    if a1 <= a0 or b1 <= b0:
        return 0.0
    before = float(np.median(x[a0:a1]))
    after = float(np.median(x[b0:b1]))
    return abs(after - before) / max(reference_p2p, EPS)


def qrs_event_times(
    env: np.ndarray,
    fs: int,
    reference_env: float,
    signal: np.ndarray | None = None,
    reference_p2p: float | None = None,
) -> np.ndarray:
    """Times at which the QRS-band envelope rises above a fraction of reference.

    One time per contiguous excursion, placed at its peak.  This is not a beat
    detector and is not used as one -- it only answers "did this record produce
    QRS-band energy here", which is what a dropout measure needs.

    When ``signal`` and ``reference_p2p`` are supplied, excursions that do not
    return to their starting baseline are discarded as non-cardiac.  Without
    that test a single electrode step inside a dropout counts as a complex,
    splits the QRS-free gap and hides the fact that the signal is gone.
    """
    if env.size == 0 or reference_env <= EPS:
        return np.asarray([], dtype=float)
    above = env >= QRS_EVENT_FRACTION * reference_env
    if not np.any(above):
        return np.asarray([], dtype=float)
    edges = np.diff(above.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if above[0]:
        starts.insert(0, 0)
    if above[-1]:
        ends.append(env.size)
    times = [
        (lo + int(np.argmax(env[lo:hi]))) / fs
        for lo, hi in zip(starts, ends)
        if hi > lo
    ]
    # A single complex can dip below the threshold mid-way and be counted twice,
    # which would shorten the record's apparent beat interval.  Merge anything
    # closer together than a refractory period.
    merged: list[float] = []
    for t in times:
        if merged and t - merged[-1] < QRS_EVENT_MERGE_SEC:
            continue
        merged.append(t)

    if signal is not None and reference_p2p is not None:
        merged = [
            t for t in merged
            if baseline_return_ratio(signal, fs, t, reference_p2p)
            <= STEP_REJECT_RATIO
        ]
    return np.asarray(merged, dtype=float)


def gap_dropout_at(
    centre: float, events: np.ndarray, threshold_sec: float, duration: float
) -> float:
    """How far into an abnormally long QRS-free gap the time ``centre`` sits."""
    if events.size == 0:
        return 1.0
    idx = int(np.searchsorted(events, centre))
    before = float(events[idx - 1]) if idx > 0 else 0.0
    after = float(events[idx]) if idx < events.size else duration
    gap = max(0.0, after - before)
    if threshold_sec <= EPS:
        return 0.0
    return float(np.clip((gap - threshold_sec) / threshold_sec, 0.0, 1.0))


def energy_dropout(ctx_energy_ratio: float) -> float:
    """Dropout implied by low absolute QRS-band energy over a beat-spanning context."""
    return float(
        np.clip(
            (ENERGY_DROPOUT_FULL - ctx_energy_ratio)
            / (ENERGY_DROPOUT_FULL - ENERGY_DROPOUT_NONE),
            0.0,
            1.0,
        )
    )


def dropout_index_at(
    centre: float,
    events: np.ndarray,
    threshold_sec: float,
    duration: float,
    ctx_energy_ratio: float,
) -> float:
    """Combine the two independent views of signal loss.

    The gap measure asks "has this record gone unusually long without a
    complex", which is immune to beats that are merely small.  The energy
    measure asks "is there any QRS-band energy here at all", which is immune to
    a lone transient -- an electrode pop, say -- splitting a long gap in two and
    thereby hiding it.  Each covers the other's blind spot, so take the worse.
    """
    return max(
        gap_dropout_at(centre, events, threshold_sec, duration),
        energy_dropout(ctx_energy_ratio),
    )


def dropout_gap_threshold(events: np.ndarray) -> float:
    """The QRS-free gap length beyond which a record is considered to have lost signal."""
    if events.size < 3:
        return DROPOUT_GAP_FLOOR_SEC
    typical = float(np.median(np.diff(events)))
    return max(DROPOUT_GAP_FLOOR_SEC, DROPOUT_GAP_MULTIPLE * typical)


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def window_features(
    seg: np.ndarray,
    fs: int,
    seg_low: np.ndarray,
    seg_env: np.ndarray,
    ctx_env: np.ndarray,
    ctx_seg: np.ndarray,
    reference_p2p: float,
    reference_env: float,
    baseline_median: float = 0.0,
    dropout_index: float = 0.0,
    reference_power: float = 0.0,
    contains_complex: bool = True,
) -> dict[str, float]:
    """Measure the quality-relevant properties of one window.

    Args:
        seg: the raw window samples.
        fs: sampling rate.
        seg_low: the corresponding sub-0.7 Hz component.
        seg_env: the 5-18 Hz envelope over exactly this window.
        ctx_env: the same envelope over a wider context centred on this window
            (see ``ENVELOPE_CONTEXT_SEC``), used only for QRS contrast.
        ctx_seg: the raw signal over that same context window, used for the
            amplitude *deficit* measure.
        reference_p2p: record-wide robust peak-to-peak reference amplitude.
        reference_env: record-wide robust QRS-band envelope peak.
        baseline_median: median of the record's sub-0.7 Hz component.
        dropout_index: gap-based dropout score for this window, computed once
            per record by :func:`dropout_index_at`.
        reference_power: record-level 0.5 Hz-Nyquist band power of a typical
            window, used to normalise interference measures.
        contains_complex: whether a QRS-band event falls inside this window.
            A window that legitimately sits between two beats is quiet by
            nature and must not be judged for lacking QRS amplitude.

    Returns:
        A dict of named, unit-free feature values.
    """
    ref = max(reference_p2p, EPS)
    n = seg.size

    # -- amplitude ----------------------------------------------------------
    # Amplitude is measured two ways, because the two directions are different
    # questions.  An *excursion* is local and transient -- motion -- so it is
    # measured on the window itself and stays sharply localised.  A *collapse*
    # is sustained -- a failing electrode -- so it is measured over a context
    # window guaranteed to span at least one beat interval.  Measuring the
    # collapse window-locally would flag two harmless situations: a window that
    # falls between beats at a slow heart rate, and a window whose boundary
    # happens to cut through a complex.
    p2p = float(np.percentile(seg, 99) - np.percentile(seg, 1))
    amplitude_ratio = p2p / ref
    ctx_p2p = float(np.percentile(ctx_seg, 99) - np.percentile(ctx_seg, 1))
    amplitude_ratio_ctx = ctx_p2p / ref

    # -- baseline drift -----------------------------------------------------
    # Two ways a baseline can be wrong: it can move *within* the window, and it
    # can sit far from where the rest of the record's baseline sits.  A slow
    # drift is almost flat near its own turning points, so the within-window
    # spread alone would report those stretches as clean.  Take the worse of
    # the two.
    within_window = float(np.std(seg_low))
    offset_from_record = abs(float(np.mean(seg_low)) - baseline_median)
    baseline_drift_ratio = max(within_window, offset_from_record) / ref

    # -- spectral content ---------------------------------------------------
    nper = min(n, int(fs))
    if nper >= 16:
        freqs, psd = sps.welch(seg - np.mean(seg), fs=fs, nperseg=nper, window="hann")
    else:  # pragma: no cover - windows are never this short in practice
        freqs, psd = np.array([0.0]), np.array([0.0])

    nyq = fs / 2.0
    total = _band_power(freqs, psd, 0.5, nyq) + EPS
    hf_hi = min(nyq, 100.0)
    hf = _band_power(freqs, psd, 40.0, hf_hi)
    # Remove the mains bands from the broadband high-frequency estimate so a
    # pure 50 Hz tone is not mistaken for muscle noise.
    pl = 0.0
    for harmonic in (1, 2):
        centre = POWERLINE_FREQ * harmonic
        if centre < nyq:
            band = _band_power(freqs, psd, centre - 2.0, centre + 2.0)
            pl += band
            if 40.0 <= centre < hf_hi:
                hf -= band
    hf = max(hf, 0.0)

    # Normalise interference against a RECORD-level in-band power reference,
    # not against this window's own total power.  At 40 BPM a 1.0 s window can
    # fall entirely between two beats; its own total power is then just the
    # sensor noise floor, so dividing by it makes an ordinary quiet stretch
    # look catastrophically noisy.  Referencing the record instead asks the
    # question that actually matters -- "is there unusual interference energy
    # here" -- and gives the same answer whether or not a complex happens to
    # sit inside the window.
    denom = max(reference_power, EPS) if reference_power > 0.0 else total
    hf_noise_ratio = hf / denom
    powerline_ratio = pl / denom

    # -- QRS visibility and dropout -----------------------------------------
    # `qrs_visibility` is a *relative* contrast measure: how far the QRS-band
    # energy rises above its own local floor.  `qrs_energy_ratio` is an
    # *absolute* one: how much QRS-band energy this window holds compared with
    # the rest of the record.  Both are needed -- a flat, disconnected
    # electrode can still show high contrast between its noise floor and a
    # stray transient, so contrast alone would not reveal that the QRS content
    # has gone.
    env_peak = float(np.percentile(seg_env, 98))
    qrs_energy_ratio = env_peak / max(reference_env, EPS)
    ctx_peak = float(np.percentile(ctx_env, 98))
    ctx_med = float(np.median(ctx_env)) + EPS
    qrs_visibility = ctx_peak / ctx_med

    # -- informational only: spectral entropy -------------------------------
    p = psd[(freqs >= 0.5) & (freqs < min(nyq, 60.0))]
    p = p / (np.sum(p) + EPS)
    spectral_entropy = (
        float(-np.sum(p * np.log(p + EPS)) / (np.log(p.size) + EPS))
        if p.size > 1
        else float("nan")
    )

    return {
        "amplitude_p2p_mv": p2p,
        "amplitude_ratio": amplitude_ratio,
        "amplitude_ratio_ctx": amplitude_ratio_ctx,
        "baseline_drift_ratio": baseline_drift_ratio,
        "hf_noise_ratio": hf_noise_ratio,
        "powerline_ratio": powerline_ratio,
        "qrs_energy_ratio": qrs_energy_ratio,
        "dropout_index": dropout_index,
        "contains_complex": float(bool(contains_complex)),
        "qrs_visibility": qrs_visibility,
        "spectral_entropy": spectral_entropy,
    }


def score_window(features: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Convert measured features into a 0-100 score plus a penalty breakdown."""
    pens: dict[str, float] = {}

    knee, sat, cap = PENALTIES["baseline_drift_ratio"]
    pens["baseline_drift"] = _linear_penalty(
        features["baseline_drift_ratio"], knee, sat, cap
    )

    knee, sat, cap = PENALTIES["hf_noise_ratio"]
    pens["high_frequency_noise"] = _linear_penalty(
        features["hf_noise_ratio"], knee, sat, cap
    )

    knee, sat, cap = PENALTIES["powerline_ratio"]
    pens["powerline_interference"] = _linear_penalty(
        features["powerline_ratio"], knee, sat, cap
    )

    knee, sat, cap = PENALTIES["amplitude_excess"]
    excess = _linear_penalty(features["amplitude_ratio"], knee, sat, cap)
    knee, sat, cap = PENALTIES["amplitude_deficit"]
    deficit = _linear_penalty(
        features.get("amplitude_ratio_ctx", features["amplitude_ratio"]),
        knee, sat, cap,
    )
    pens["amplitude_anomaly"] = max(excess, deficit)

    knee, sat, cap = PENALTIES["dropout_index"]
    pens["dropout"] = _linear_penalty(features["dropout_index"], knee, sat, cap)

    # QRS visibility is a "higher is better" feature: convert it to a deficit.
    vis = features["qrs_visibility"]
    deficit_frac = (QRS_VISIBILITY_GOOD - vis) / (
        QRS_VISIBILITY_GOOD - QRS_VISIBILITY_BAD
    )
    _, _, cap = PENALTIES["qrs_visibility_deficit"]
    pens["qrs_visibility"] = float(np.clip(deficit_frac, 0.0, 1.0) * cap)

    score = float(np.clip(100.0 - sum(pens.values()), 0.0, 100.0))
    return score, pens


def _level(value: float, low: float, high: float, invert: bool = False) -> str:
    """Map a number onto a low/moderate/high verbal level for the dashboard.

    With ``invert=True`` a *large* input means a *low* reported level, which is
    how penalties are turned into stability statements.
    """
    if np.isnan(value):
        return "unknown"
    if invert:
        if value >= high:
            return "low"
        if value >= low:
            return "moderate"
        return "high"
    if value >= high:
        return "high"
    if value >= low:
        return "moderate"
    return "low"


def record_band_power(
    x: np.ndarray, fs: int, window_sec: float = QUALITY_WINDOW_SEC
) -> float:
    """Typical in-band power of one window of this record.

    The median over windows, so a few loud or silent stretches do not set the
    scale.  Used to normalise the interference measures, which would otherwise
    be divided by whatever happened to be inside a single window.
    """
    w = max(8, int(window_sec * fs))
    n_win = max(1, x.size // w)
    nyq = fs / 2.0
    vals = []
    for i in range(n_win):
        seg = x[i * w:(i + 1) * w]
        if seg.size < 16:
            continue
        freqs, psd = sps.welch(
            seg - np.mean(seg), fs=fs, nperseg=min(seg.size, int(fs)), window="hann"
        )
        vals.append(_band_power(freqs, psd, 0.5, nyq))
    return float(np.median(vals)) if vals else 0.0


def record_references(
    x: np.ndarray,
    env: np.ndarray,
    fs: int,
    window_sec: float = QUALITY_WINDOW_SEC,
) -> tuple[float, float]:
    """Estimate this record's normal QRS amplitude and QRS-band energy.

    Both references are taken from the windows in which a QRS is *most clearly
    visible*, judged by QRS-band contrast -- a self-normalised ratio that needs
    no reference of its own, so there is no circularity.

    Taking a plain median over all windows fails in two opposite ways.  If most
    of the record is dropped out, the median describes the dropout, and the few
    intact stretches then measure as huge amplitude excursions.  If part of the
    record has large motion excursions, those inflate it instead.  Selecting
    the better-contrast half of the record avoids both, because motion destroys
    contrast and dropout destroys energy.

    Returns:
        ``(reference_p2p, reference_env_peak)``.
    """
    w = max(8, int(window_sec * fs))
    n_win = max(1, min(x.size, env.size) // w)
    p2p, peak, contrast = [], [], []
    for i in range(n_win):
        seg = x[i * w:(i + 1) * w]
        eseg = env[i * w:(i + 1) * w]
        if seg.size < 4 or eseg.size < 4:
            continue
        ep = float(np.percentile(eseg, 98))
        em = float(np.median(eseg)) + EPS
        p2p.append(float(np.percentile(seg, 99) - np.percentile(seg, 1)))
        peak.append(ep)
        contrast.append(ep / em)

    if not p2p:
        return (float(np.ptp(x)) or 1.0, float(np.max(env)) or 1.0)

    p2p_a = np.asarray(p2p)
    peak_a = np.asarray(peak)
    contrast_a = np.asarray(contrast)

    if p2p_a.size >= 4:
        keep = contrast_a >= np.median(contrast_a)
        if not np.any(keep):
            keep = np.ones_like(contrast_a, dtype=bool)
    else:
        keep = np.ones_like(contrast_a, dtype=bool)

    ref_p2p = float(np.median(p2p_a[keep])) or float(np.median(p2p_a)) or 1.0
    ref_env = float(np.median(peak_a[keep])) or float(np.median(peak_a)) or 1.0
    return ref_p2p, ref_env


def reference_amplitude(
    x: np.ndarray, fs: int, window_sec: float = QUALITY_WINDOW_SEC
) -> float:
    """Convenience wrapper returning only the peak-to-peak reference."""
    return record_references(x, _bandpass_envelope(x, fs), fs, window_sec)[0]


def assess_quality(
    x: np.ndarray,
    fs: int,
    window_sec: float = QUALITY_WINDOW_SEC,
    hop_sec: float = QUALITY_HOP_SEC,
    reference_p2p: float | None = None,
) -> QualityReport:
    """Compute the Prototype Signal Quality Index for a signal.

    Args:
        x: the waveform.
        fs: sampling rate in Hz.
        window_sec: analysis window length.
        hop_sec: step between successive windows.
        reference_p2p: optional externally supplied amplitude reference.  Pass
            the raw signal's reference when scoring a processed signal so the
            two scores are directly comparable.

    Returns:
        A :class:`QualityReport`.
    """
    x = np.asarray(x, dtype=float)
    if x.size < fs:  # shorter than one second
        raise ValueError("signal too short for quality assessment")

    low = _lowfreq_component(x, fs)
    env = _bandpass_envelope(x, fs)
    ref_auto, ref_env = record_references(x, env, fs, window_sec)
    ref = ref_auto if reference_p2p is None else reference_p2p
    baseline_median = float(np.median(low))
    events = qrs_event_times(env, fs, ref_env, x, ref)
    ref_power = record_band_power(x, fs, window_sec)
    gap_threshold = dropout_gap_threshold(events)
    duration = x.size / fs
    # Context wide enough to always span at least one beat interval, so a slow
    # rhythm is never mistaken for missing signal.
    typical_gap = float(np.median(np.diff(events))) if events.size >= 3 else 0.8
    context_sec = max(ENVELOPE_CONTEXT_SEC, 1.3 * typical_gap)

    w = int(window_sec * fs)
    hop = max(1, int(hop_sec * fs))
    ctx_pad = max(0, int((context_sec * fs - w) / 2))
    windows: list[WindowQuality] = []
    idx = 0
    start = 0
    while start + w <= x.size:
        seg = x[start:start + w]
        clo = max(0, start - ctx_pad)
        chi = min(x.size, start + w + ctx_pad)
        centre = (start + 0.5 * w) / fs
        ctx_energy = float(np.percentile(env[clo:chi], 98)) / max(ref_env, EPS)
        w_lo, w_hi = start / fs, (start + w) / fs
        has_complex = bool(np.any((events >= w_lo) & (events < w_hi)))
        feats = window_features(
            seg, fs, low[start:start + w], env[start:start + w], env[clo:chi],
            x[clo:chi], ref, ref_env, baseline_median,
            dropout_index_at(centre, events, gap_threshold, duration, ctx_energy),
            ref_power, has_complex,
        )
        score, pens = score_window(feats)
        windows.append(
            WindowQuality(
                index=idx,
                start=start / fs,
                end=(start + w) / fs,
                score=score,
                features=feats,
                penalties=pens,
            )
        )
        idx += 1
        start += hop

    scores = np.asarray([wq.score for wq in windows], dtype=float)
    overall = float(np.mean(scores)) if scores.size else 0.0

    def _mean_pen(key: str) -> float:
        return float(np.mean([wq.penalties[key] for wq in windows])) if windows else 0.0

    def _mean_feat(key: str) -> float:
        return (
            float(np.mean([wq.features[key] for wq in windows]))
            if windows
            else float("nan")
        )

    artifact_burden = float(np.mean(scores < QUALITY_TRUSTED_MIN)) if scores.size else 0.0

    evidence: dict[str, Any] = {
        "score": round(overall, 1),
        "worst_window_score": round(float(np.min(scores)), 1) if scores.size else 0.0,
        "best_window_score": round(float(np.max(scores)), 1) if scores.size else 0.0,
        "n_windows": len(windows),
        "baseline_stability": _level(_mean_pen("baseline_drift"), 8.0, 25.0, invert=True),
        "qrs_visibility": _level(
            _mean_feat("qrs_visibility"), QRS_VISIBILITY_BAD, QRS_VISIBILITY_GOOD
        ),
        "high_frequency_noise": _level(_mean_pen("high_frequency_noise"), 8.0, 22.0),
        "powerline_interference": _level(
            _mean_pen("powerline_interference"), 6.0, 18.0
        ),
        "amplitude_stability": _level(
            _mean_pen("amplitude_anomaly"), 8.0, 25.0, invert=True
        ),
        "dropout": _level(_mean_pen("dropout"), 5.0, 20.0),
        "artifact_burden": _level(artifact_burden, 0.08, 0.30),
        "artifact_burden_fraction": round(artifact_burden, 3),
        "mean_penalties": {
            k: round(_mean_pen(k), 1)
            for k in (
                "baseline_drift",
                "high_frequency_noise",
                "powerline_interference",
                "amplitude_anomaly",
                "dropout",
                "qrs_visibility",
            )
        },
        "mean_features": {
            "baseline_drift_ratio": round(_mean_feat("baseline_drift_ratio"), 4),
            "hf_noise_ratio": round(_mean_feat("hf_noise_ratio"), 4),
            "powerline_ratio": round(_mean_feat("powerline_ratio"), 4),
            "amplitude_ratio": round(_mean_feat("amplitude_ratio"), 3),
            "dropout_index": round(_mean_feat("dropout_index"), 3),
            "qrs_visibility": round(_mean_feat("qrs_visibility"), 2),
            "spectral_entropy": round(_mean_feat("spectral_entropy"), 3),
        },
    }

    return QualityReport(
        score=overall, windows=windows, evidence=evidence, reference_p2p=ref
    )
