"""Shared configuration constants for ECG Guardian.

Everything tunable by the pipeline lives here so that thresholds are explicit,
auditable and never buried inside an algorithm.
"""

from __future__ import annotations

# ---------------------------------------------------------------- acquisition
DEFAULT_SAMPLING_RATE: int = 250          # Hz
DEFAULT_DURATION: float = 30.0            # seconds
DEFAULT_HEART_RATE: float = 72.0          # BPM
DEFAULT_SEED: int = 42

POWERLINE_FREQ: float = 50.0              # Hz (regional mains frequency)

# ------------------------------------------------------------- quality engine
QUALITY_WINDOW_SEC: float = 1.0           # analysis window for windowed quality
QUALITY_HOP_SEC: float = 0.5              # hop between windows (50% overlap)

# Prototype Signal Quality Index band edges (0-100)
QUALITY_TRUSTED_MIN: float = 70.0         # >= this -> window is trustworthy
QUALITY_DEGRADED_MIN: float = 45.0        # >= this -> degraded but maybe usable
# below QUALITY_DEGRADED_MIN -> unusable unless recovery succeeds

# -------------------------------------------------------------- recovery gate
# A recovery attempt is only "VALIDATED" when both are satisfied:
RECOVERY_MIN_ABSOLUTE_GAIN: float = 8.0   # quality points gained
RECOVERY_MIN_POST_SCORE: float = 55.0     # post-recovery score floor

# --------------------------------------------------------- R-peak detection
REFRACTORY_SEC: float = 0.24              # minimum spacing between R peaks
QRS_BAND: tuple[float, float] = (5.0, 18.0)   # Hz, energy band for detection
INTEGRATION_WINDOW_SEC: float = 0.12      # moving-window integration length

# Physiologically plausible bounds used only as sanity rails (not diagnosis)
RR_MIN_SEC: float = 0.30                  # 200 BPM ceiling
RR_MAX_SEC: float = 2.00                  # 30 BPM floor

# ------------------------------------------------- measurement evidence gates
# Heart rate
HR_MIN_VALID_BEATS: int = 8
HR_MIN_VALID_BEAT_FRACTION: float = 0.55
HR_MIN_RR_CONSISTENCY: float = 0.70
HR_MIN_LOCAL_QUALITY: float = 55.0
HR_MIN_CONFIDENCE: float = 0.60

# RR interval
RR_MIN_VALID_INTERVALS: int = 6
RR_MIN_CONSISTENCY: float = 0.65
RR_MIN_CONFIDENCE: float = 0.60

# RR consistency (reported as its own measurement)
RRC_MIN_VALID_INTERVALS: int = 8
RRC_MIN_CONFIDENCE: float = 0.55

# QRS duration -- deliberately stricter: needs morphology, not just rhythm
QRS_MIN_VALID_BEATS: int = 6
QRS_MIN_BEAT_FRACTION: float = 0.60
QRS_MIN_LOCAL_QUALITY: float = 72.0
QRS_MAX_DISPERSION_MS: float = 14.0       # IQR across per-beat estimates
QRS_PLAUSIBLE_MS: tuple[float, float] = (55.0, 145.0)
QRS_MIN_CONFIDENCE: float = 0.65
QRS_MAX_HF_NOISE: float = 0.35            # normalised high-frequency burden

# ------------------------------------------------------- beat-level standards
# The heart of measurement-specific gating: a beat can be good enough to say
# *when* it happened without being good enough to say *how wide* it was.  So
# the same beat is judged against a different standard depending on what the
# measurement needs from it.
BEAT_STANDARD_TIMING: dict[str, object] = {
    "label": "timing evidence",
    "min_local_quality": HR_MIN_LOCAL_QUALITY,
    "min_relative_prominence": 0.25,
    "amplitude_range": (0.40, 2.50),
    "min_template_correlation": 0.70,
}

BEAT_STANDARD_MORPHOLOGY: dict[str, object] = {
    "label": "morphological evidence",
    "min_local_quality": QRS_MIN_LOCAL_QUALITY,
    "min_relative_prominence": 0.55,
    "amplitude_range": (0.70, 1.60),
    "min_template_correlation": 0.90,
    "max_hf_noise_ratio": QRS_MAX_HF_NOISE,
    "min_slope_contrast": 3.0,
}

# ------------------------------------------------------------------ labelling
TRUST_TRUSTED = "TRUSTED"
TRUST_DEGRADED = "DEGRADED"
TRUST_RECOVERED = "RECOVERED"
TRUST_REJECTED = "REJECTED"

DISCLAIMER = (
    "Prototype engineering system. Not intended for clinical diagnosis."
)
