"""Adversarial regression tests.

Every test in this file corresponds to a specific way the system was found to
be foolable, or a specific assumption it was found to be making.  They are
written as invariants rather than as fixed numbers, so they keep their meaning
if the implementation changes.

The organising principle: a measurement may be refused at any time, but a
measurement that is *reported* must be right.  False acceptance is the failure
mode that matters; conservative rejection is not a bug.
"""

import numpy as np
import pytest

from src.adversarial import (
    QRS_ERROR_TOLERANCE_MS,
    expectation_for,
    inject_dropout_over_beats,
    inject_flatline,
    inject_inverted_segment,
    inject_spike_train,
    inject_step,
    reference_qrs_ms,
)
from src.artifacts import corrupt_ecg, corrupt_many
from src.config import TRUST_REJECTED, TRUST_TRUSTED
from src.detection import detect_artifacts
from src.ecg_generator import generate_ecg
from src.evidence import ACCEPTED
from src.filtering import recover
from src.pipeline import run_pipeline
from src.quality import (
    STEP_REJECT_RATIO,
    _bandpass_envelope,
    assess_quality,
    baseline_return_ratio,
    qrs_event_times,
    record_references,
)

HR_TOL = 5.0


def hr_of(result):
    m = result.measurement("heart_rate")
    return m.value if m and m.status == ACCEPTED else None


def qrs_of(result):
    m = result.measurement("qrs_duration")
    return m.value if m and m.status == ACCEPTED else None


# ===========================================================================
# REGRESSION: a step discontinuity must not pass as a complex
# ===========================================================================
# Found: at contact-loss severity 0.95 the injected step deposited enough
# broadband energy to register as a QRS event, which split the QRS-free gap in
# two and hid the dropout, so the region's recovery was wrongly revalidated.
# Root cause: the dropout measure asked only "is there QRS-band energy here",
# a question a step answers just as well as a complex does.
# Fix: a complex is a transient that returns to baseline; a step is a level
# change that does not.  `baseline_return_ratio` measures exactly that.

def test_step_and_complex_are_separated_by_baseline_return():
    """The discriminator must separate the two classes with real margin."""
    rec = generate_ecg(seed=42)
    fs = rec.sampling_rate
    env = _bandpass_envelope(rec.signal, fs)
    ref_p2p, ref_env = record_references(rec.signal, env, fs)

    beats = qrs_event_times(env, fs, ref_env)
    beat_ratios = [baseline_return_ratio(rec.signal, fs, t, ref_p2p) for t in beats]
    assert max(beat_ratios) < STEP_REJECT_RATIO, (
        "real complexes must return to baseline"
    )

    stepped = inject_step(rec, 12.0, 1.0)
    at_step = baseline_return_ratio(stepped.signal, fs, 12.0, ref_p2p)
    assert at_step > STEP_REJECT_RATIO, "a step must not look like a complex"
    assert at_step > 2.0 * max(beat_ratios), "the margin must not be marginal"


@pytest.mark.parametrize("hr", [40, 72, 120, 180])
def test_real_beats_return_to_baseline_at_every_rate(hr):
    rec = generate_ecg(heart_rate=hr, seed=hr)
    fs = rec.sampling_rate
    env = _bandpass_envelope(rec.signal, fs)
    ref_p2p, ref_env = record_references(rec.signal, env, fs)
    ratios = [
        baseline_return_ratio(rec.signal, fs, t, ref_p2p)
        for t in qrs_event_times(env, fs, ref_env)
    ]
    assert max(ratios) < STEP_REJECT_RATIO


def test_baseline_wander_is_not_mistaken_for_a_step():
    """Drift moves the baseline slowly; the discriminator must not fire on it."""
    rec = corrupt_ecg(generate_ecg(seed=42), "baseline_wander", severity=0.95, seed=1)
    fs = rec.sampling_rate
    env = _bandpass_envelope(rec.signal, fs)
    ref_p2p, ref_env = record_references(rec.signal, env, fs)
    ratios = [
        baseline_return_ratio(rec.signal, fs, t, ref_p2p)
        for t in qrs_event_times(env, fs, ref_env)
    ]
    assert max(ratios) < STEP_REJECT_RATIO


@pytest.mark.parametrize("severity", [0.6, 0.8, 0.95, 1.0])
def test_contact_loss_is_never_declared_recovered(severity):
    """The headline regression: severe contact loss must fail revalidation."""
    rec = corrupt_ecg(
        generate_ecg(seed=42), "electrode_contact",
        start=8.0, duration=3.0, severity=severity, seed=7,
    )
    q = assess_quality(rec.signal, rec.sampling_rate)
    dets = detect_artifacts(q, duration=rec.duration)
    rep = recover(rec.signal, rec.sampling_rate, dets, q)
    assert rep.status != "VALIDATED", (
        f"severity {severity}: filtering cannot restore absent QRS content"
    )


def test_contact_loss_recovery_is_monotonic_in_severity():
    """Worse damage must never be easier to recover than lesser damage."""
    validated = []
    for severity in (0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        rec = corrupt_ecg(
            generate_ecg(seed=42), "electrode_contact",
            start=8.0, duration=3.0, severity=severity, seed=7,
        )
        q = assess_quality(rec.signal, rec.sampling_rate)
        dets = detect_artifacts(q, duration=rec.duration)
        validated.append(recover(rec.signal, rec.sampling_rate, dets, q).status
                         == "VALIDATED")
    assert not any(validated), f"none of these should revalidate: {validated}"


# ===========================================================================
# REGRESSION: beat-free windows at slow heart rates
# ===========================================================================
# Found: a clean 40 BPM record was labelled REJECTED.  At 40 BPM the beats are
# 1.5 s apart, so a 1.0 s window can contain no complex at all.  Its own total
# power is then just the noise floor, so normalising interference by it made an
# ordinary quiet stretch look catastrophically noisy, and its peak-to-peak was
# tiny so the amplitude-deficit penalty fired as well.

@pytest.mark.parametrize("hr", [34, 38, 40, 45, 48, 55])
def test_slow_clean_records_are_trusted(hr):
    result = run_pipeline(generate_ecg(heart_rate=hr, seed=hr), f"hr{hr}")
    assert result.detections == [], f"{hr} BPM: false artifact detection"
    assert result.overall_trust == TRUST_TRUSTED
    assert all(s.label == TRUST_TRUSTED for s in result.trust_map)


@pytest.mark.parametrize("hr", [34, 40, 48, 60, 100, 150, 180])
def test_heart_rate_is_accurate_across_the_rate_range(hr):
    rec = generate_ecg(heart_rate=hr, seed=hr)
    result = run_pipeline(rec, f"hr{hr}")
    value = hr_of(result)
    assert value is not None, f"{hr} BPM was refused on a clean record"
    assert abs(value - rec.metadata["true_mean_hr_bpm"]) < HR_TOL


def test_interference_measures_ignore_whether_a_window_holds_a_beat():
    """A quiet inter-beat window must not read as interference."""
    rec = generate_ecg(heart_rate=40, seed=40)
    q = assess_quality(rec.signal, rec.sampling_rate)
    for w in q.windows:
        assert w.penalties["high_frequency_noise"] < 5.0, f"at {w.start}s"
        assert w.penalties["powerline_interference"] < 5.0, f"at {w.start}s"
        assert w.penalties["amplitude_anomaly"] < 5.0, f"at {w.start}s"


# ===========================================================================
# REGRESSION: sampling-rate assumptions
# ===========================================================================
# Found: a clean 360 Hz record refused QRS duration with a beat-to-beat spread
# of 18.8 ms.  The boundary search required the low-slope state to persist for
# `int(0.008 * fs)` samples, which truncates to 5.6 ms at 360 Hz instead of the
# intended 8 ms, so a momentary dip was taken for the end of the complex.

@pytest.mark.parametrize("fs", [200, 250, 300, 360, 400, 500])
def test_qrs_duration_is_stable_across_sampling_rates(fs):
    rec = generate_ecg(sampling_rate=fs, seed=11)
    result = run_pipeline(rec, f"fs{fs}")
    value = qrs_of(result)
    assert value is not None, f"fs={fs}: refused on a clean record"
    assert abs(value - reference_qrs_ms(rec)) < QRS_ERROR_TOLERANCE_MS


@pytest.mark.parametrize("fs", [200, 250, 360, 500])
def test_clean_records_are_trusted_at_every_sampling_rate(fs):
    result = run_pipeline(generate_ecg(sampling_rate=fs, seed=11), f"fs{fs}")
    assert result.detections == []
    assert result.overall_trust == TRUST_TRUSTED


# ===========================================================================
# PHASE 4/5: attacking the gate and the confidence
# ===========================================================================
def test_periodic_non_cardiac_spikes_do_not_produce_a_heart_rate():
    """The confidence attack.

    Regularly spaced spikes on a flat trace give perfectly consistent RR
    intervals.  Rhythm regularity must not be accepted as evidence of cardiac
    origin.
    """
    rec = inject_spike_train(
        inject_flatline(generate_ecg(seed=42), 5.0, 20.0),
        5.2, 19.0, period=0.83,
    )
    result = run_pipeline(rec, "spikes")
    assert hr_of(result) is None, "heart rate reported from non-cardiac spikes"
    assert qrs_of(result) is None
    hr = result.measurement("heart_rate")
    assert hr.confidence < 0.6
    assert hr.reason


def test_flatline_region_is_never_trusted():
    rec = inject_flatline(generate_ecg(seed=42), 8.0, 8.0)
    result = run_pipeline(rec, "flat")
    covering = [s for s in result.trust_map if s.start <= 12.0 <= s.end]
    assert covering and covering[0].label == TRUST_REJECTED


def test_a_step_does_not_make_a_dropout_look_occupied():
    """A flatline with a step in the middle must still read as a dropout."""
    rec = inject_flatline(inject_step(generate_ecg(seed=42), 9.0, 2.0), 9.0, 6.0)
    result = run_pipeline(rec, "flat_step")
    covering = [s for s in result.trust_map if s.start <= 12.0 <= s.end]
    assert covering and covering[0].label == TRUST_REJECTED
    hr = result.measurement("heart_rate")
    inside = [
        v for v in hr.beat_verdicts if 9.5 <= v.time <= 14.5
    ]
    assert all(not v.accepted for v in inside), "beats used from inside a dropout"


def test_beats_dropped_out_are_not_used():
    dropped = list(range(10, 18))
    rec = inject_dropout_over_beats(generate_ecg(seed=42), dropped)
    result = run_pipeline(rec, "dropped")
    hr = result.measurement("heart_rate")
    gone = {float(rec.r_peaks_true[i]) / rec.sampling_rate for i in dropped}
    for v in hr.beat_verdicts:
        if any(abs(v.time - g) < 0.12 for g in gone):
            assert not v.accepted, f"used a removed beat at {v.time:.2f}s"


def test_measurements_survive_a_polarity_inversion_safely():
    """Either refuse, or report a correct number.  Never report a wrong one."""
    rec = inject_inverted_segment(generate_ecg(seed=42), 10.0, 6.0)
    result = run_pipeline(rec, "inverted")
    value = hr_of(result)
    if value is not None:
        assert abs(value - rec.metadata["true_mean_hr_bpm"]) < HR_TOL


def test_morphology_damage_alone_refuses_only_the_morphology_measurement():
    """Timing survives heavy broadband noise; onset/offset does not."""
    rec = corrupt_ecg(generate_ecg(seed=42), "muscle_noise", severity=0.95, seed=25)
    result = run_pipeline(rec, "emg")
    assert hr_of(result) is not None, "timing evidence should have survived"
    assert qrs_of(result) is None, "morphology evidence should not have"


def test_confidence_is_low_wherever_a_measurement_is_refused():
    """A refusal must not carry a confident-looking number alongside it."""
    for builder, key in (
        (lambda: inject_flatline(generate_ecg(seed=42), 3.0, 24.0), "flat"),
        (lambda: inject_spike_train(
            inject_flatline(generate_ecg(seed=42), 4.0, 22.0), 4.3, 21.0, 0.75
        ), "spikes"),
    ):
        result = run_pipeline(builder(), key)
        for m in result.measurements:
            if m.status != ACCEPTED:
                assert m.value is None
                assert m.confidence < 0.85, f"{key}/{m.measurement}"


# ===========================================================================
# PHASE 3: cross-artifact combinations
# ===========================================================================
COMBOS = [
    ("baseline+muscle", [
        {"artifact_type": "baseline_wander", "severity": 0.8},
        {"artifact_type": "muscle_noise", "severity": 0.6},
    ]),
    ("contact+motion", [
        {"artifact_type": "electrode_contact", "start": 6.0, "duration": 5.0,
         "severity": 0.9},
        {"artifact_type": "motion", "start": 18.0, "duration": 3.0, "severity": 0.8},
    ]),
    ("motion+powerline", [
        {"artifact_type": "motion", "start": 12.0, "duration": 4.0, "severity": 0.8},
        {"artifact_type": "powerline", "severity": 0.7},
    ]),
    ("everything_severe", [
        {"artifact_type": "baseline_wander", "severity": 0.9},
        {"artifact_type": "muscle_noise", "severity": 0.8},
        {"artifact_type": "motion", "start": 7.0, "duration": 5.0, "severity": 0.9},
        {"artifact_type": "electrode_contact", "start": 18.0, "duration": 6.0,
         "severity": 0.95},
    ]),
]


@pytest.mark.parametrize("name,specs", COMBOS, ids=[c[0] for c in COMBOS])
def test_combined_artifacts_never_yield_a_wrong_number(name, specs):
    rec = corrupt_many(generate_ecg(seed=42), specs)
    result = run_pipeline(rec, name)
    assert result.errors == []
    value = hr_of(result)
    if value is not None:
        assert abs(value - rec.metadata["true_mean_hr_bpm"]) < HR_TOL, name
    qrs = qrs_of(result)
    if qrs is not None:
        assert abs(qrs - reference_qrs_ms(rec)) < QRS_ERROR_TOLERANCE_MS, name


# ===========================================================================
# PHASE 6: recovery must earn its result
# ===========================================================================
def test_quality_improvement_alone_never_validates_a_recovery(results_cache={}):
    """A rising score is not evidence that the missing content came back."""
    rec = corrupt_ecg(
        generate_ecg(seed=42), "electrode_contact",
        start=8.0, duration=3.0, severity=0.95, seed=7,
    )
    q = assess_quality(rec.signal, rec.sampling_rate)
    dets = detect_artifacts(q, duration=rec.duration)
    rep = recover(rec.signal, rec.sampling_rate, dets, q)
    assert rep.regions
    assert rep.affected_after > rep.affected_before, (
        "this case is only meaningful while filtering does raise the score"
    )
    assert rep.status != "VALIDATED"
    assert any("did not normalise" in r.reason for r in rep.regions)


# ===========================================================================
# PHASE 8: randomised property testing
# ===========================================================================
@pytest.mark.parametrize("seed", [11, 23, 47, 91, 137])
def test_random_scenarios_never_report_a_wrong_heart_rate(seed):
    """Deterministic random sweep.  The seed reproduces any failure exactly."""
    rng = np.random.default_rng(seed)
    for _ in range(6):
        hr = float(rng.uniform(40, 150))
        duration = float(rng.choice([15.0, 30.0]))
        specs = []
        for _ in range(int(rng.integers(0, 3))):
            span = float(rng.uniform(1.0, duration * 0.6))
            specs.append({
                "artifact_type": str(rng.choice([
                    "baseline_wander", "powerline", "muscle_noise", "motion",
                    "electrode_contact",
                ])),
                "start": float(rng.uniform(0.0, max(0.1, duration - span))),
                "duration": span,
                "severity": float(rng.uniform(0.2, 1.0)),
                "seed": int(rng.integers(0, 9999)),
            })
        base = generate_ecg(heart_rate=hr, duration=duration,
                            seed=int(rng.integers(0, 9999)))
        rec = corrupt_many(base, specs) if specs else base
        result = run_pipeline(rec, "rand")
        assert result.errors == []
        value = hr_of(result)
        if value is not None:
            err = abs(value - rec.metadata["true_mean_hr_bpm"])
            assert err < HR_TOL, (
                f"seed={seed} hr={hr:.1f} specs={specs} -> reported {value} "
                f"(true {rec.metadata['true_mean_hr_bpm']:.1f})"
            )


@pytest.mark.parametrize("seed", [3, 19, 64])
def test_random_scenarios_produce_no_nans_or_crashes(seed):
    rng = np.random.default_rng(seed)
    for _ in range(6):
        duration = float(rng.choice([8.0, 20.0, 40.0]))
        base = generate_ecg(
            heart_rate=float(rng.uniform(35, 170)), duration=duration,
            sampling_rate=int(rng.choice([200, 250, 360])),
            seed=int(rng.integers(0, 9999)),
        )
        result = run_pipeline(base, "rand")
        assert result.errors == []
        assert result.trust_map
        assert np.all(np.isfinite(result.processed_signal))
        for m in result.measurements:
            assert np.isfinite(m.confidence)
            if m.value is not None:
                assert np.isfinite(m.value)


# ===========================================================================
# Ground-truth machinery sanity
# ===========================================================================
def test_expectation_treats_a_dc_step_as_harmless_to_timing():
    """The harness itself must not call a level shift 'damage'."""
    clean = generate_ecg(seed=42)
    stepped = inject_step(clean, 12.0, 2.0)
    exp = expectation_for(stepped, clean)
    assert exp.hr_supportable
    assert exp.timed_beats > 0.9 * exp.n_beats


def test_expectation_knows_a_flatline_destroys_beats():
    """A flatline must count as lost beats, and enough of them to matter.

    Note the harness counts surviving beats rather than a surviving fraction:
    20 s of flatline in a 30 s record still leaves 12 intact beats, which do
    genuinely support a heart rate.  It takes a longer outage to make the
    measurement unsupportable.
    """
    clean = generate_ecg(seed=42)

    partial = expectation_for(inject_flatline(clean, 5.0, 20.0), clean)
    assert partial.timed_beats < 0.5 * partial.n_beats
    assert partial.hr_supportable, "12 surviving beats still support a rate"

    total = expectation_for(inject_flatline(clean, 2.0, 26.0), clean)
    assert not total.hr_supportable
    assert not total.qrs_supportable
