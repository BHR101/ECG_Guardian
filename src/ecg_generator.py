"""Phase 1 -- deterministic synthetic single-lead ECG generator.

The waveform is built from a sum of Gaussian components (P, Q, R, S, T) placed
relative to each R-peak.  This is a well-known modelling approach; nothing here
is claimed as novel.  Its only job is to give the rest of the pipeline a signal
whose ground truth we actually know, so that validation is honest.

Ground truth (R-peak sample indices, per-beat onsets/offsets) is returned for
validation and demo purposes ONLY.  The detection and quality code must never
import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    DEFAULT_DURATION,
    DEFAULT_HEART_RATE,
    DEFAULT_SAMPLING_RATE,
    DEFAULT_SEED,
)

# Gaussian wave components, expressed relative to the R-peak instant.
# (name, amplitude in mV, centre offset in s, sigma in s)
_WAVE_COMPONENTS: tuple[tuple[str, float, float, float], ...] = (
    ("P", 0.130, -0.200, 0.0250),
    ("Q", -0.140, -0.030, 0.0085),
    ("R", 1.000, 0.000, 0.0120),
    ("S", -0.230, 0.034, 0.0110),
    ("T", 0.300, 0.220, 0.0460),
)

# Nominal ground-truth QRS window relative to the R peak (seconds).
# Derived from the Q and S component positions/widths above.
TRUE_QRS_ONSET_OFFSET: tuple[float, float] = (-0.052, 0.058)
TRUE_QRS_DURATION_MS: float = (
    TRUE_QRS_ONSET_OFFSET[1] - TRUE_QRS_ONSET_OFFSET[0]
) * 1000.0


@dataclass
class ECGRecord:
    """A generated ECG together with its ground truth and metadata."""

    time: np.ndarray
    signal: np.ndarray
    sampling_rate: int
    r_peaks_true: np.ndarray            # sample indices -- VALIDATION ONLY
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(len(self.signal)) / self.sampling_rate

    def copy_with(self, signal: np.ndarray, **meta: Any) -> "ECGRecord":
        """Return a new record sharing the ground truth but a different signal."""
        merged = dict(self.metadata)
        merged.update(meta)
        return ECGRecord(
            time=self.time,
            signal=signal,
            sampling_rate=self.sampling_rate,
            r_peaks_true=self.r_peaks_true,
            metadata=merged,
        )


def _rr_series(
    duration: float,
    heart_rate: float,
    rng: np.random.Generator,
    variability: float,
) -> np.ndarray:
    """Build a sequence of RR intervals with mild physiological variability.

    Two contributions are used: a slow respiratory-style sinusoidal modulation
    and a small amount of beat-to-beat jitter.  Both are seeded, so the result
    is reproducible.
    """
    mean_rr = 60.0 / float(heart_rate)
    intervals: list[float] = []
    elapsed = 0.0
    beat = 0
    # Generate a little past the requested duration so the tail is populated.
    while elapsed < duration + 2.0 * mean_rr:
        respiratory = 1.0 + variability * np.sin(2.0 * np.pi * 0.25 * elapsed)
        jitter = 1.0 + variability * 0.35 * rng.standard_normal()
        rr = mean_rr * respiratory * jitter
        rr = float(np.clip(rr, 0.4 * mean_rr, 1.8 * mean_rr))
        intervals.append(rr)
        elapsed += rr
        beat += 1
    return np.asarray(intervals, dtype=float)


def generate_ecg(
    duration: float = DEFAULT_DURATION,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    heart_rate: float = DEFAULT_HEART_RATE,
    seed: int = DEFAULT_SEED,
    variability: float = 0.030,
    sensor_noise: float = 0.006,
) -> ECGRecord:
    """Generate a plausible synthetic single-lead ECG.

    Args:
        duration: record length in seconds.
        sampling_rate: samples per second.
        heart_rate: mean heart rate in BPM.
        seed: RNG seed -- identical seeds give byte-identical signals.
        variability: fractional RR modulation depth (mild, physiological).
        sensor_noise: standard deviation of the always-present sensor noise
            floor, in mV.  A perfectly noiseless signal is unrealistic and
            makes spectral measures degenerate.

    Returns:
        An :class:`ECGRecord` with time, signal, ground-truth R-peak indices
        and metadata.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not 20.0 <= heart_rate <= 220.0:
        raise ValueError("heart_rate must be between 20 and 220 BPM")

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration * sampling_rate))
    time = np.arange(n_samples, dtype=float) / sampling_rate
    signal = np.zeros(n_samples, dtype=float)

    intervals = _rr_series(duration, heart_rate, rng, variability)

    # Place beats.  The first R peak is offset so the first P wave fits.
    r_times: list[float] = []
    t_beat = 0.35
    for rr in intervals:
        if t_beat >= duration:
            break
        r_times.append(t_beat)
        t_beat += rr

    # Per-beat amplitude scaling: small, so morphology stays consistent.
    for r_time in r_times:
        beat_scale = 1.0 + 0.02 * rng.standard_normal()
        for _name, amp, centre, sigma in _WAVE_COMPONENTS:
            mu = r_time + centre
            # Only evaluate the Gaussian where it is non-negligible (+-4 sigma).
            lo = max(0, int((mu - 4.0 * sigma) * sampling_rate))
            hi = min(n_samples, int((mu + 4.0 * sigma) * sampling_rate) + 1)
            if hi <= lo:
                continue
            seg_t = time[lo:hi]
            signal[lo:hi] += (
                amp * beat_scale
                * np.exp(-0.5 * ((seg_t - mu) / sigma) ** 2)
            )

    # Always-present sensor noise floor.
    signal = signal + sensor_noise * rng.standard_normal(n_samples)

    r_peaks_true = np.asarray(
        [int(round(t * sampling_rate)) for t in r_times], dtype=int
    )
    r_peaks_true = r_peaks_true[
        (r_peaks_true >= 0) & (r_peaks_true < n_samples)
    ]

    rr_true = np.diff(r_peaks_true) / sampling_rate if len(r_peaks_true) > 1 else np.array([])

    generator_kwargs = {
        "duration": float(duration),
        "sampling_rate": int(sampling_rate),
        "heart_rate": float(heart_rate),
        "seed": int(seed),
        "variability": float(variability),
    }

    metadata: dict[str, Any] = {
        "source": "synthetic",
        "generator_kwargs": generator_kwargs,
        "duration_s": float(n_samples) / sampling_rate,
        "sampling_rate": sampling_rate,
        "requested_heart_rate_bpm": float(heart_rate),
        "true_mean_hr_bpm": float(60.0 / np.mean(rr_true)) if rr_true.size else float("nan"),
        "true_beat_count": int(r_peaks_true.size),
        "true_qrs_duration_ms": TRUE_QRS_DURATION_MS,
        "seed": int(seed),
        "variability": float(variability),
        "sensor_noise_mv": float(sensor_noise),
        "artifacts": [],
    }

    return ECGRecord(
        time=time,
        signal=signal,
        sampling_rate=sampling_rate,
        r_peaks_true=r_peaks_true,
        metadata=metadata,
    )


def noise_free_twin(record: ECGRecord) -> ECGRecord:
    """Regenerate ``record`` with the sensor noise floor set to zero.

    The RNG is consumed in the same order either way, so the beat positions are
    identical -- only the additive noise differs.  Used during validation to
    obtain a reference QRS duration measured by the *same* estimator under
    ideal conditions, which is the only fair way to separate estimator bias
    from the effect of noise and artifacts.

    Raises:
        ValueError: if the record was not produced by :func:`generate_ecg`.
    """
    kwargs = record.metadata.get("generator_kwargs")
    if not kwargs:
        raise ValueError("record has no generator parameters; cannot rebuild it")
    return generate_ecg(sensor_noise=0.0, **kwargs)
