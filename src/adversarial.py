"""Adversarial test corpus.

The goal of this module is not to make ECG Guardian look good.  It is to find
cases where the system reports a measurement it does not have the evidence to
support.

Two independent notions of "the evidence was insufficient" are used, because
the interesting one must not be circular:

1. **Outcome-based (primary).**  We know the true heart rate and we can measure
   the true QRS width on a noise-free twin of every record.  If the system
   reports a number that is materially wrong, the evidence was insufficient --
   no argument about thresholds is possible.  This is the metric that matters.

2. **Evidence-based (secondary).**  By comparing each beat in the corrupted
   record against the same beat in its clean twin, we can say objectively how
   much the artifact distorted it.  That gives a per-record expectation of
   which measurements *ought* to be supportable, derived from the signals
   themselves rather than from the artifact's label or the system's own
   feature thresholds.

Everything here is ground-truth machinery.  It is used for scenario
construction and scoring only, and nothing in it is reachable from the
analysis path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from .artifacts import corrupt_ecg, corrupt_many
from .ecg_generator import ECGRecord, generate_ecg, noise_free_twin

# --------------------------------------------------------------------------
# How much a beat may be disturbed before we stop expecting it to be usable.
# These bound the *ground truth*, not the system: they describe the waveform,
# not any feature the pipeline computes.
# --------------------------------------------------------------------------
QRS_HALF_WIDTH_SEC = 0.06      # window treated as "the complex" for scoring
TIMING_TOLERANCE_SEC = 0.03    # peak may move this far and still time the beat
MORPHOLOGY_MAX_DISTORTION = 0.35   # relative L2 change over the QRS window
TIMING_MAX_DISTORTION = 1.20       # beyond this the complex is unrecognisable

# Minimum surviving evidence for a measurement to be *expected* to work.
MIN_TIMED_BEATS = 10
MIN_TIMED_FRACTION = 0.60    # diagnostic only, not a gate
MIN_MORPH_BEATS = 8
MIN_MORPH_FRACTION = 0.65    # diagnostic only, not a gate

# Tolerances for judging a reported value materially wrong.
HR_ERROR_TOLERANCE_BPM = 5.0
QRS_ERROR_TOLERANCE_MS = 20.0


# ==========================================================================
# Adversarial waveform primitives
# ==========================================================================
# These are corpus-construction tools, deliberately kept out of the production
# artifact taxonomy in src/artifacts.py.  They exist to attack the system with
# shapes a well-behaved artifact model would not produce.

def _ref_amplitude(record: ECGRecord) -> float:
    return float(np.percentile(np.abs(record.signal), 99.5)) or 1.0


def _note(record: ECGRecord, kind: str, start: float, end: float,
          severity: float, recoverable: bool, notes: str) -> None:
    record.metadata.setdefault("artifacts", [])
    record.metadata["artifacts"].append(
        {
            "type": kind, "start": float(start), "end": float(end),
            "severity": float(severity), "recoverable": recoverable,
            "notes": notes,
        }
    )


def inject_step(record: ECGRecord, at: float, height: float = 1.0) -> ECGRecord:
    """A pure, instantaneous DC level shift that never returns.

    The single most important adversarial shape: broadband edge energy that
    looks like a complex to any energy-only measure, but carries no cardiac
    information whatsoever.
    """
    fs = record.sampling_rate
    x = record.signal.copy()
    idx = int(np.clip(at * fs, 0, x.size - 1))
    x[idx:] += height * _ref_amplitude(record)
    out = record.copy_with(x)
    _note(out, "step_discontinuity", at, record.duration, min(1.0, abs(height)),
          False, "instantaneous DC level shift, no return to baseline")
    return out


def inject_flatline(record: ECGRecord, start: float, duration: float) -> ECGRecord:
    """Total signal loss: the trace is replaced by exactly nothing."""
    fs = record.sampling_rate
    x = record.signal.copy()
    lo = int(np.clip(start * fs, 0, x.size))
    hi = int(np.clip((start + duration) * fs, 0, x.size))
    level = float(np.median(x[max(0, lo - fs):lo])) if lo > 0 else 0.0
    x[lo:hi] = level
    out = record.copy_with(x)
    _note(out, "flatline", start, start + duration, 1.0, False,
          "signal replaced by a constant level")
    return out


def inject_spike_train(
    record: ECGRecord,
    start: float,
    duration: float,
    period: float = 0.8,
    amplitude: float = 1.0,
    width_ms: float = 12.0,
) -> ECGRecord:
    """Regularly spaced non-cardiac spikes.

    The confidence attack.  These produce peaks with entirely plausible,
    highly consistent RR intervals, so any heart-rate logic that trusts
    rhythm regularity as evidence of cardiac origin will accept them.
    """
    fs = record.sampling_rate
    x = record.signal.copy()
    ref = _ref_amplitude(record)
    half = max(1, int(width_ms * 1e-3 * fs / 2))
    t = start
    while t < start + duration:
        idx = int(t * fs)
        lo, hi = max(0, idx - half), min(x.size, idx + half + 1)
        if hi > lo:
            # A narrow triangular spike: no P wave, no T wave, no return arc.
            ramp = 1.0 - np.abs(np.linspace(-1.0, 1.0, hi - lo))
            x[lo:hi] += amplitude * ref * ramp
        t += period
    out = record.copy_with(x)
    _note(out, "spike_train", start, start + duration, min(1.0, amplitude),
          False, f"non-cardiac spikes every {period:.2f} s")
    return out


def inject_dropout_over_beats(
    record: ECGRecord, beat_indices: list[int], pad_sec: float = 0.12,
    severity: float = 1.0,
) -> ECGRecord:
    """Remove the signal exactly over chosen beats.

    Uses the ground-truth beat positions -- legitimate here, because this is
    scenario construction, not analysis.
    """
    fs = record.sampling_rate
    x = record.signal.copy()
    spans = []
    for i in beat_indices:
        if not 0 <= i < record.r_peaks_true.size:
            continue
        centre = int(record.r_peaks_true[i])
        lo = max(0, centre - int(pad_sec * fs))
        hi = min(x.size, centre + int(pad_sec * fs))
        level = float(np.median(x[max(0, lo - fs // 4):lo])) if lo > 0 else 0.0
        x[lo:hi] = (1.0 - severity) * x[lo:hi] + severity * level
        spans.append((lo / fs, hi / fs))
    out = record.copy_with(x)
    for lo, hi in spans:
        _note(out, "beat_dropout", lo, hi, severity, False,
              "signal removed over a complex")
    return out


def inject_repeated_dropout(
    record: ECGRecord, start: float, n: int, on: float, off: float,
    severity: float = 0.95,
) -> ECGRecord:
    """Intermittent contact: several separated dropout regions."""
    out = record
    t = start
    for _ in range(n):
        out = corrupt_ecg(out, "electrode_contact", start=t, duration=on,
                          severity=severity, seed=int(t * 37) % 1000)
        t += on + off
    return out


def inject_inverted_segment(record: ECGRecord, start: float, duration: float) -> ECGRecord:
    """Polarity flip -- a lead reversal or a wiring fault, not a rhythm change."""
    fs = record.sampling_rate
    x = record.signal.copy()
    lo = int(np.clip(start * fs, 0, x.size))
    hi = int(np.clip((start + duration) * fs, 0, x.size))
    x[lo:hi] = -x[lo:hi]
    out = record.copy_with(x)
    _note(out, "polarity_inversion", start, start + duration, 1.0, False,
          "segment polarity inverted")
    return out


# ==========================================================================
# Ground-truth expectation, derived from the waveforms themselves
# ==========================================================================
@dataclass
class BeatTruth:
    """How badly the artifact actually damaged one beat."""

    index: int
    time: float
    distortion: float          # relative L2 change over the QRS window
    peak_shift_sec: float      # how far the window maximum moved
    timing_intact: bool
    morphology_intact: bool


@dataclass
class Expectation:
    """What this record can legitimately support, judged from the signals."""

    n_beats: int
    timed_beats: int
    morph_beats: int
    hr_supportable: bool
    qrs_supportable: bool
    beats: list[BeatTruth] = field(default_factory=list)

    @property
    def timed_fraction(self) -> float:
        return self.timed_beats / max(1, self.n_beats)

    @property
    def morph_fraction(self) -> float:
        return self.morph_beats / max(1, self.n_beats)


def _reference_filter(x: np.ndarray, fs: int) -> np.ndarray:
    """A standard, non-adaptive cleaning chain used only for scoring.

    The ground-truth question is "did the cardiac information survive", not
    "did it survive *unfiltered*".  A narrowband mains tone sitting on top of a
    complex changes the raw waveform a great deal while destroying nothing --
    any notch filter recovers it exactly.  Judging morphology only on the raw
    corrupted trace would therefore mark recoverable beats as destroyed and
    report the system's correct recoveries as false acceptances.

    This chain is fixed and knows nothing about the artifact or the pipeline's
    own decisions; it just establishes an upper bound on what was recoverable.
    """
    from .filtering import highpass, lowpass, notch

    try:
        return lowpass(notch(highpass(x, fs, cutoff=0.5), fs), fs, cutoff=35.0)
    except Exception:  # pragma: no cover - degenerate input
        return x


def beat_truth(corrupt: ECGRecord, clean: ECGRecord) -> list[BeatTruth]:
    """Compare every beat against the same beat in the clean twin.

    Distortion is the *smaller* of the raw comparison and the comparison after
    a standard filter chain, so a beat only counts as damaged when no ordinary
    filtering could have brought it back.
    """
    fs = corrupt.sampling_rate
    half = int(QRS_HALF_WIDTH_SEC * fs)
    clean_f = _reference_filter(clean.signal, fs)
    corrupt_f = _reference_filter(corrupt.signal, fs)
    out: list[BeatTruth] = []
    for i, idx in enumerate(clean.r_peaks_true):
        lo, hi = int(idx) - half, int(idx) + half + 1
        if lo < 0 or hi > clean.signal.size:
            continue
        # Mean-centre both windows before comparing.  A constant level shift
        # (a DC step, or a slow baseline wander passing through) changes the
        # raw L2 distance enormously while destroying no information at all --
        # it is exactly what a high-pass filter legitimately removes.  What
        # matters is whether the *shape* of the complex survived.
        def _distort(ref_raw: np.ndarray, got_raw: np.ndarray) -> tuple[float, float]:
            # Shape distance, invariant to both offset and scale.  A uniform
            # amplitude change destroys neither the timing of a complex nor its
            # width -- a QRS duration is a length, and lengths do not care how
            # tall the complex is -- so scaling must not count as damage.
            ref_seg = ref_raw - np.mean(ref_raw)
            got_seg = got_raw - np.mean(got_raw)
            ref_n = float(np.linalg.norm(ref_seg))
            got_n = float(np.linalg.norm(got_seg))
            if ref_n < 1e-12:
                return 0.0, 0.0
            if got_n < 1e-12:
                return 2.0, QRS_HALF_WIDTH_SEC   # nothing left at all
            d = float(np.linalg.norm(got_seg / got_n - ref_seg / ref_n))
            sft = abs(int(np.argmax(got_seg)) - int(np.argmax(ref_seg))) / fs
            return d, sft

        d_raw, s_raw = _distort(clean.signal[lo:hi], corrupt.signal[lo:hi])
        d_filt, s_filt = _distort(clean_f[lo:hi], corrupt_f[lo:hi])
        distortion = min(d_raw, d_filt)
        shift = s_raw if d_raw <= d_filt else s_filt
        out.append(
            BeatTruth(
                index=i,
                time=float(idx) / fs,
                distortion=distortion,
                peak_shift_sec=shift,
                timing_intact=(
                    distortion <= TIMING_MAX_DISTORTION
                    and shift <= TIMING_TOLERANCE_SEC
                ),
                morphology_intact=distortion <= MORPHOLOGY_MAX_DISTORTION,
            )
        )
    return out


def expectation_for(corrupt: ECGRecord, clean: ECGRecord) -> Expectation:
    """Decide what this record ought to be able to support."""
    beats = beat_truth(corrupt, clean)
    n = len(beats)
    timed = sum(1 for b in beats if b.timing_intact)
    morph = sum(1 for b in beats if b.morphology_intact)
    return Expectation(
        n_beats=n,
        timed_beats=timed,
        morph_beats=morph,
        # Count, not fraction.  ECG Guardian reports a measurement from the
        # beats it validated and names the regions that supported it, so a
        # record with 14 intact beats out of 36 genuinely does support a heart
        # rate -- requiring a *proportion* of the record to survive would be
        # asking the wrong question.  The fractions stay as diagnostics.
        # A record cannot be asked for more beats than it physically contains:
        # 12 s at 39 BPM holds eight, and eight intact beats do support a rate.
        hr_supportable=timed >= max(6, min(MIN_TIMED_BEATS, n)),
        qrs_supportable=morph >= max(5, min(MIN_MORPH_BEATS, n)),
        beats=beats,
    )


# ==========================================================================
# Cases
# ==========================================================================
@dataclass
class AdversarialCase:
    """One adversarial record plus what it ought to support."""

    key: str
    family: str
    record: ECGRecord
    clean: ECGRecord
    params: dict[str, Any] = field(default_factory=dict)

    _expectation: Expectation | None = None

    @property
    def expectation(self) -> Expectation:
        if self._expectation is None:
            self._expectation = expectation_for(self.record, self.clean)
        return self._expectation


def _case(key: str, family: str, build: Callable[[ECGRecord], ECGRecord],
          base_kwargs: dict[str, Any] | None = None,
          **params: Any) -> AdversarialCase:
    kwargs = base_kwargs or {}
    clean = generate_ecg(**kwargs)
    record = build(clean)
    return AdversarialCase(
        key=key, family=family, record=record, clean=clean, params=params
    )


# -------------------------------------------------------------- A. clean ---
def clean_cases() -> Iterator[AdversarialCase]:
    for hr in (40, 48, 60, 72, 95, 120, 150):
        yield _case(f"clean_hr{hr}", "clean", lambda r: r,
                    {"heart_rate": hr, "seed": hr}, heart_rate=hr)
    for var in (0.005, 0.03, 0.08, 0.14):
        yield _case(f"clean_var{var}", "clean", lambda r: r,
                    {"variability": var, "seed": 5}, variability=var)
    for dur in (8.0, 15.0, 30.0, 60.0):
        yield _case(f"clean_dur{dur:.0f}", "clean", lambda r: r,
                    {"duration": dur, "seed": 9}, duration=dur)
    for seed in (1, 2, 3, 4, 5):
        yield _case(f"clean_seed{seed}", "clean", lambda r: r,
                    {"seed": seed}, seed=seed)
    for fs in (200, 250, 360, 500):
        yield _case(f"clean_fs{fs}", "clean", lambda r: r,
                    {"sampling_rate": fs, "seed": 11}, sampling_rate=fs)


# ------------------------------------------- B-E. single-artifact sweeps ---
def single_artifact_cases() -> Iterator[AdversarialCase]:
    families = {
        "baseline_wander": "baseline",
        "powerline": "powerline",
        "muscle_noise": "muscle",
        "motion": "motion",
        "electrode_contact": "contact",
    }
    for artifact, family in families.items():
        for sev in (0.3, 0.5, 0.7, 0.9, 1.0):
            # localised
            yield _case(
                f"{artifact}_local_s{sev}", family,
                lambda r, a=artifact, s=sev: corrupt_ecg(
                    r, a, start=8.0, duration=3.0, severity=s, seed=21
                ),
                {"seed": 42}, artifact=artifact, severity=sev, scope="local",
            )
            # global
            yield _case(
                f"{artifact}_global_s{sev}", family,
                lambda r, a=artifact, s=sev: corrupt_ecg(
                    r, a, severity=s, seed=22
                ),
                {"seed": 42}, artifact=artifact, severity=sev, scope="global",
            )
        # very short and very long spans
        for dur in (0.4, 1.0, 12.0, 24.0):
            yield _case(
                f"{artifact}_dur{dur}", family,
                lambda r, a=artifact, d=dur: corrupt_ecg(
                    r, a, start=4.0, duration=d, severity=0.85, seed=23
                ),
                {"seed": 42}, artifact=artifact, span=dur,
            )


# ----------------------------------------- F. contact loss, the main target ---
def contact_loss_cases() -> Iterator[AdversarialCase]:
    yield _case("step_mid", "contact",
                lambda r: inject_step(r, 12.0, 1.0), {"seed": 42})
    yield _case("step_large", "contact",
                lambda r: inject_step(r, 12.0, 3.0), {"seed": 42})
    yield _case("step_small", "contact",
                lambda r: inject_step(r, 12.0, 0.4), {"seed": 42})
    yield _case("step_pair", "contact",
                lambda r: inject_step(inject_step(r, 10.0, 2.0), 16.0, -2.0),
                {"seed": 42})
    yield _case("flatline_3s", "contact",
                lambda r: inject_flatline(r, 9.0, 3.0), {"seed": 42})
    yield _case("flatline_12s", "contact",
                lambda r: inject_flatline(r, 6.0, 12.0), {"seed": 42})
    yield _case("flatline_with_step", "contact",
                lambda r: inject_flatline(inject_step(r, 9.0, 2.0), 9.0, 4.0),
                {"seed": 42})
    yield _case("dropout_over_1_beat", "contact",
                lambda r: inject_dropout_over_beats(r, [10]), {"seed": 42})
    yield _case("dropout_over_5_beats", "contact",
                lambda r: inject_dropout_over_beats(r, list(range(10, 15))),
                {"seed": 42})
    yield _case("dropout_over_12_beats", "contact",
                lambda r: inject_dropout_over_beats(r, list(range(8, 20))),
                {"seed": 42})
    yield _case("dropout_alternating", "contact",
                lambda r: inject_dropout_over_beats(r, list(range(6, 30, 2))),
                {"seed": 42})
    yield _case("dropout_between_beats", "contact",
                lambda r: inject_flatline(r, 9.42, 0.45), {"seed": 42})
    yield _case("repeated_dropout", "contact",
                lambda r: inject_repeated_dropout(r, 4.0, 4, 2.0, 2.5),
                {"seed": 42})
    yield _case("gradual_degradation", "contact",
                lambda r: corrupt_many(r, [
                    {"artifact_type": "electrode_contact", "start": 5.0,
                     "duration": 6.0, "severity": 0.35},
                    {"artifact_type": "electrode_contact", "start": 11.0,
                     "duration": 6.0, "severity": 0.65},
                    {"artifact_type": "electrode_contact", "start": 17.0,
                     "duration": 8.0, "severity": 0.95},
                ]), {"seed": 42})
    yield _case("contact_severe_long", "contact",
                lambda r: corrupt_ecg(r, "electrode_contact", start=3.0,
                                      duration=22.0, severity=0.98, seed=24),
                {"seed": 42})
    yield _case("polarity_flip", "contact",
                lambda r: inject_inverted_segment(r, 10.0, 6.0), {"seed": 42})


# ---------------------------------- Phase 4/5: gate and confidence attacks ---
def gate_attack_cases() -> Iterator[AdversarialCase]:
    # Periodic non-cardiac spikes with perfectly plausible RR intervals.
    yield _case("spikes_over_flatline", "attack",
                lambda r: inject_spike_train(
                    inject_flatline(r, 5.0, 20.0), 5.2, 19.0, period=0.83
                ), {"seed": 42})
    yield _case("spikes_plausible_rr", "attack",
                lambda r: inject_spike_train(
                    inject_flatline(r, 4.0, 22.0), 4.3, 21.0, period=0.75
                ), {"seed": 42})
    yield _case("spikes_added_to_signal", "attack",
                lambda r: inject_spike_train(r, 5.0, 20.0, period=0.41),
                {"seed": 42})
    # Morphology destroyed, timing preserved.
    yield _case("morphology_only_damage", "attack",
                lambda r: corrupt_ecg(r, "muscle_noise", severity=0.95, seed=25),
                {"seed": 42})
    # Timing damaged, morphology locally fine.
    yield _case("few_good_beats", "attack",
                lambda r: inject_dropout_over_beats(
                    r, [i for i in range(36) if i not in (2, 3, 4, 20, 21)]
                ), {"seed": 42})
    yield _case("one_critical_region", "attack",
                lambda r: corrupt_ecg(r, "motion", start=14.0, duration=3.0,
                                      severity=0.95, seed=26), {"seed": 42})
    yield _case("mostly_good_one_bad", "attack",
                lambda r: inject_flatline(r, 14.0, 2.0), {"seed": 42})
    yield _case("tiny_trustworthy_portion", "attack",
                lambda r: corrupt_ecg(r, "motion", start=0.0, duration=26.0,
                                      severity=0.95, seed=27), {"seed": 42})


# --------------------------------------------- Phase 3: cross-combinations ---
def combination_cases() -> Iterator[AdversarialCase]:
    combos: list[tuple[str, list[dict[str, Any]]]] = [
        ("baseline_muscle", [
            {"artifact_type": "baseline_wander", "severity": 0.8},
            {"artifact_type": "muscle_noise", "severity": 0.6},
        ]),
        ("baseline_motion", [
            {"artifact_type": "baseline_wander", "severity": 0.7},
            {"artifact_type": "motion", "start": 10.0, "duration": 3.0,
             "severity": 0.8},
        ]),
        ("motion_muscle", [
            {"artifact_type": "motion", "start": 8.0, "duration": 3.0,
             "severity": 0.7},
            {"artifact_type": "muscle_noise", "severity": 0.5},
        ]),
        ("powerline_muscle", [
            {"artifact_type": "powerline", "severity": 0.8},
            {"artifact_type": "muscle_noise", "severity": 0.6},
        ]),
        ("contact_baseline", [
            {"artifact_type": "electrode_contact", "start": 9.0,
             "duration": 5.0, "severity": 0.9},
            {"artifact_type": "baseline_wander", "severity": 0.7},
        ]),
        ("contact_motion", [
            {"artifact_type": "electrode_contact", "start": 6.0,
             "duration": 5.0, "severity": 0.9},
            {"artifact_type": "motion", "start": 18.0, "duration": 3.0,
             "severity": 0.8},
        ]),
        ("contact_muscle", [
            {"artifact_type": "electrode_contact", "start": 8.0,
             "duration": 6.0, "severity": 0.9},
            {"artifact_type": "muscle_noise", "severity": 0.6},
        ]),
        ("motion_powerline", [
            {"artifact_type": "motion", "start": 12.0, "duration": 4.0,
             "severity": 0.8},
            {"artifact_type": "powerline", "severity": 0.7},
        ]),
        ("everything_mild", [
            {"artifact_type": "baseline_wander", "severity": 0.4},
            {"artifact_type": "powerline", "severity": 0.35},
            {"artifact_type": "muscle_noise", "severity": 0.3},
        ]),
        ("everything_severe", [
            {"artifact_type": "baseline_wander", "severity": 0.9},
            {"artifact_type": "muscle_noise", "severity": 0.8},
            {"artifact_type": "motion", "start": 7.0, "duration": 5.0,
             "severity": 0.9},
            {"artifact_type": "electrode_contact", "start": 18.0,
             "duration": 6.0, "severity": 0.95},
        ]),
    ]
    for name, specs in combos:
        yield _case(f"combo_{name}", "combination",
                    lambda r, s=specs: corrupt_many(r, s), {"seed": 42},
                    combo=name)


# ----------------------------------------- Phase 8: randomised exploration ---
ARTIFACTS = ("baseline_wander", "powerline", "muscle_noise", "motion",
             "electrode_contact")


def random_cases(n: int = 200, seed: int = 20240501) -> Iterator[AdversarialCase]:
    """Deterministic randomised scenarios.  The seed is the reproduction key."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        case_seed = int(rng.integers(0, 10_000))
        hr = float(rng.uniform(38, 150))
        duration = float(rng.choice([12.0, 20.0, 30.0, 45.0]))
        variability = float(rng.uniform(0.005, 0.12))
        n_art = int(rng.integers(0, 4))
        specs = []
        for _ in range(n_art):
            artifact = str(rng.choice(ARTIFACTS))
            span = float(rng.uniform(0.5, duration * 0.8))
            start = float(rng.uniform(0.0, max(0.1, duration - span)))
            specs.append({
                "artifact_type": artifact,
                "start": start,
                "duration": span,
                "severity": float(rng.uniform(0.2, 1.0)),
                "seed": int(rng.integers(0, 10_000)),
            })
        params = {
            "heart_rate": hr, "duration": duration,
            "variability": variability, "seed": case_seed,
            "artifacts": specs,
        }
        yield _case(
            f"rand_{i:03d}", "random",
            lambda r, s=specs: corrupt_many(r, s) if s else r,
            {"heart_rate": hr, "duration": duration,
             "variability": variability, "seed": case_seed},
            **params,
        )


def all_cases(include_random: int = 200) -> Iterator[AdversarialCase]:
    yield from clean_cases()
    yield from single_artifact_cases()
    yield from contact_loss_cases()
    yield from gate_attack_cases()
    yield from combination_cases()
    if include_random:
        yield from random_cases(include_random)


# ==========================================================================
# Scoring
# ==========================================================================
TRUE_ACCEPT = "TRUE_ACCEPT"
FALSE_ACCEPT = "FALSE_ACCEPT"
TRUE_REJECT = "TRUE_REJECT"
FALSE_REJECT = "FALSE_REJECT"


def reference_qrs_ms(clean: ECGRecord) -> float:
    """QRS width measured by our own estimator on the noise-free twin."""
    from .measurements import measure_morphology
    from .peaks import detect_r_peaks

    try:
        twin = noise_free_twin(clean)
    except ValueError:
        twin = clean
    report = detect_r_peaks(twin.signal, twin.sampling_rate)
    morph = measure_morphology(twin.signal, twin.sampling_rate, report.beats)
    widths = [m.duration_ms for m in morph if m.clear]
    return float(np.median(widths)) if widths else float("nan")


def judge_case(case: AdversarialCase, result: Any) -> dict[str, Any]:
    """Score one analysed case.

    Returns a flat row: the outcome-based verdict (was a reported number
    actually right?) and the evidence-based verdict (should it have been
    reportable at all?), for heart rate and for QRS duration.
    """
    exp = case.expectation
    true_hr = case.clean.metadata.get("true_mean_hr_bpm", float("nan"))
    ref_qrs = reference_qrs_ms(case.clean)

    row: dict[str, Any] = {
        "case": case.key,
        "family": case.family,
        "n_true_beats": exp.n_beats,
        "timed_beats": exp.timed_beats,
        "morph_beats": exp.morph_beats,
        "timed_fraction": round(exp.timed_fraction, 3),
        "morph_fraction": round(exp.morph_fraction, 3),
        "hr_supportable": exp.hr_supportable,
        "qrs_supportable": exp.qrs_supportable,
        "overall_trust": getattr(result, "overall_trust", "?"),
        "recovery_status": result.recovery.status,
        "pipeline_errors": len(result.errors),
    }

    for name, truth_ok, true_value, tol, key in (
        ("heart_rate", exp.hr_supportable, true_hr, HR_ERROR_TOLERANCE_BPM, "hr"),
        ("qrs_duration", exp.qrs_supportable, ref_qrs, QRS_ERROR_TOLERANCE_MS, "qrs"),
    ):
        m = result.measurement(name)
        reported = m is not None and m.status == "ACCEPTED" and m.value is not None
        conf = m.confidence if m is not None else float("nan")
        err = (
            abs(m.value - true_value)
            if reported and not np.isnan(true_value)
            else float("nan")
        )
        # Outcome verdict: a reported number that is materially wrong is an
        # unsupported measurement, whatever the evidence bookkeeping said.
        if reported:
            outcome = FALSE_ACCEPT if (np.isnan(err) or err > tol) else TRUE_ACCEPT
        else:
            outcome = TRUE_REJECT if not truth_ok else FALSE_REJECT
        # Evidence verdict: compare against what the waveform could support.
        if reported and not truth_ok:
            evidence = FALSE_ACCEPT
        elif reported and truth_ok:
            evidence = TRUE_ACCEPT
        elif not reported and truth_ok:
            evidence = FALSE_REJECT
        else:
            evidence = TRUE_REJECT

        row[f"{key}_reported"] = reported
        row[f"{key}_value"] = m.value if m is not None else None
        row[f"{key}_true"] = round(float(true_value), 2) if not np.isnan(true_value) else None
        row[f"{key}_error"] = round(float(err), 3) if not np.isnan(err) else None
        row[f"{key}_confidence"] = round(float(conf), 3) if not np.isnan(conf) else None
        row[f"{key}_outcome"] = outcome
        row[f"{key}_evidence_verdict"] = evidence
        row[f"{key}_reason"] = m.reason if m is not None else ""

    return row
