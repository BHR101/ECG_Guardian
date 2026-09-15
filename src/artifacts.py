"""Phase 2 -- controlled artifact injection with ground-truth metadata.

Each injector corrupts a *known* region of the record and records exactly what
it did.  That ground truth is used to score artifact localisation during
validation; it is never handed to the detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import signal as sps

from .config import POWERLINE_FREQ
from .ecg_generator import ECGRecord

ArtifactType = Literal[
    "baseline_wander",
    "powerline",
    "muscle_noise",
    "motion",
    "electrode_contact",
]

ARTIFACT_TYPES: tuple[str, ...] = (
    "baseline_wander",
    "powerline",
    "muscle_noise",
    "motion",
    "electrode_contact",
)

HUMAN_LABELS: dict[str, str] = {
    "baseline_wander": "Baseline wander",
    "powerline": f"{int(POWERLINE_FREQ)} Hz power-line interference",
    "muscle_noise": "Muscle / high-frequency noise",
    "motion": "Motion artifact",
    "electrode_contact": "Electrode / contact degradation",
    "unknown": "Unknown / uncertain disturbance",
    "none": "No artifact",
}


@dataclass
class ArtifactSpec:
    """Ground-truth description of one injected artifact."""

    type: str
    start: float
    end: float
    severity: float
    recoverable: bool
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _region_mask(
    n: int, sampling_rate: int, start: float, end: float
) -> tuple[int, int]:
    lo = int(np.clip(round(start * sampling_rate), 0, n))
    hi = int(np.clip(round(end * sampling_rate), 0, n))
    if hi <= lo:
        hi = min(n, lo + 1)
    return lo, hi


def _taper(length: int, fraction: float = 0.12) -> np.ndarray:
    """Raised-cosine envelope so injected regions start and stop smoothly."""
    if length <= 2:
        return np.ones(length)
    ramp = max(1, int(length * fraction))
    env = np.ones(length)
    ramp_curve = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    env[:ramp] = ramp_curve
    env[-ramp:] = ramp_curve[::-1]
    return env


def corrupt_ecg(
    record: ECGRecord,
    artifact_type: str,
    start: float | None = None,
    duration: float | None = None,
    severity: float = 0.6,
    seed: int = 0,
) -> ECGRecord:
    """Inject one artifact into a copy of ``record``.

    Args:
        record: source ECG (not modified).
        artifact_type: one of :data:`ARTIFACT_TYPES`.
        start: artifact start time in seconds; ``None`` means the whole record.
        duration: artifact length in seconds; ``None`` means to the end.
        severity: 0..1 scaling of the disturbance magnitude.
        seed: RNG seed for the stochastic components.

    Returns:
        A new :class:`ECGRecord` whose metadata gains an ``artifacts`` entry.
    """
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(
            f"unknown artifact_type {artifact_type!r}; "
            f"expected one of {ARTIFACT_TYPES}"
        )
    severity = float(np.clip(severity, 0.0, 1.0))

    fs = record.sampling_rate
    x = np.array(record.signal, dtype=float, copy=True)
    n = x.size
    rng = np.random.default_rng(seed)

    t_start = 0.0 if start is None else float(start)
    t_end = record.duration if duration is None else t_start + float(duration)
    lo, hi = _region_mask(n, fs, t_start, t_end)
    seg_len = hi - lo
    seg_t = record.time[lo:hi]
    env = _taper(seg_len)

    # Reference amplitude: how big the QRS complexes are in this record.
    ref = float(np.percentile(np.abs(record.signal), 99.5)) or 1.0

    recoverable = True
    notes = ""

    if artifact_type == "baseline_wander":
        wander = (
            1.00 * np.sin(2.0 * np.pi * 0.15 * seg_t + 0.7)
            + 0.55 * np.sin(2.0 * np.pi * 0.33 * seg_t + 2.1)
            + 0.30 * np.sin(2.0 * np.pi * 0.05 * seg_t)
        )
        x[lo:hi] += severity * 0.85 * ref * wander * env
        notes = "low-frequency drift below 0.5 Hz"

    elif artifact_type == "powerline":
        mains = np.sin(2.0 * np.pi * POWERLINE_FREQ * seg_t)
        harmonic = 0.25 * np.sin(2.0 * np.pi * 2 * POWERLINE_FREQ * seg_t)
        x[lo:hi] += severity * 0.30 * ref * (mains + harmonic) * env
        notes = f"narrowband interference at {POWERLINE_FREQ:.0f} Hz"

    elif artifact_type == "muscle_noise":
        raw = rng.standard_normal(seg_len)
        nyq = fs / 2.0
        high = min(0.95 * nyq, 95.0)
        sos = sps.butter(4, [20.0 / nyq, high / nyq], btype="band", output="sos")
        emg = sps.sosfiltfilt(sos, raw) if seg_len > 30 else raw
        emg = emg / (np.std(emg) or 1.0)
        x[lo:hi] += severity * 0.32 * ref * emg * env
        notes = "broadband EMG-like energy above 20 Hz"

    elif artifact_type == "motion":
        # Motion = large low-frequency excursion + a burst of mid-band energy.
        excursion = (
            np.sin(2.0 * np.pi * 0.8 * (seg_t - seg_t[0]))
            + 0.6 * np.sin(2.0 * np.pi * 2.3 * (seg_t - seg_t[0]) + 1.1)
        )
        raw = rng.standard_normal(seg_len)
        nyq = fs / 2.0
        sos = sps.butter(4, [4.0 / nyq, 30.0 / nyq], btype="band", output="sos")
        burst = sps.sosfiltfilt(sos, raw) if seg_len > 30 else raw
        burst = burst / (np.std(burst) or 1.0)
        x[lo:hi] += severity * ref * (1.45 * excursion + 0.55 * burst) * env
        notes = "abrupt localised excursion with mid-band energy"
        recoverable = severity <= 0.85

    elif artifact_type == "electrode_contact":
        # Contact loss: the QRS content disappears, leaving a drifting,
        # near-flat trace with occasional step discontinuities.
        drift = 0.12 * ref * np.sin(2.0 * np.pi * 0.2 * seg_t)
        residual = (1.0 - severity) * x[lo:hi]
        noise = 0.02 * ref * rng.standard_normal(seg_len)
        replaced = residual + severity * (drift + noise)
        if seg_len > 10:
            step_at = seg_len // 3
            replaced[step_at:] += severity * 0.9 * ref
        x[lo:hi] = (1.0 - env) * x[lo:hi] + env * replaced
        notes = "loss of QRS content, step discontinuity and flat trace"
        recoverable = severity < 0.5

    spec = ArtifactSpec(
        type=artifact_type,
        start=float(lo) / fs,
        end=float(hi) / fs,
        severity=severity,
        recoverable=recoverable,
        notes=notes,
    )

    new_record = record.copy_with(x)
    new_record.metadata["artifacts"] = list(
        record.metadata.get("artifacts", [])
    ) + [spec.as_dict()]
    return new_record


def corrupt_many(
    record: ECGRecord, specs: list[dict[str, Any]]
) -> ECGRecord:
    """Apply several artifacts in sequence.

    Each dict in ``specs`` is forwarded as keyword arguments to
    :func:`corrupt_ecg`.
    """
    out = record
    for i, spec in enumerate(specs):
        kwargs = dict(spec)
        kwargs.setdefault("seed", 100 + i)
        out = corrupt_ecg(out, **kwargs)
    return out
