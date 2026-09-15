"""Tests for the quality engine, artifact detection and recovery/revalidation."""

import numpy as np
import pytest

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg
from src.config import QUALITY_TRUSTED_MIN
from src.detection import detect_artifacts
from src.ecg_generator import generate_ecg
from src.filtering import highpass, lowpass, notch, recover
from src.quality import (
    QRS_VISIBILITY_GOOD,
    assess_quality,
    energy_dropout,
    qrs_event_times,
    record_references,
    score_window,
)


@pytest.fixture(scope="module")
def clean():
    return generate_ecg(seed=42)


@pytest.fixture(scope="module")
def clean_quality(clean):
    return assess_quality(clean.signal, clean.sampling_rate)


# ---------------------------------------------------------------- quality ---
def test_clean_signal_scores_highly(clean_quality):
    assert clean_quality.score > 90
    assert clean_quality.evidence["worst_window_score"] > 70


def test_score_is_bounded(clean_quality):
    assert np.all(clean_quality.scores >= 0)
    assert np.all(clean_quality.scores <= 100)


def test_windows_tile_the_record(clean, clean_quality):
    assert len(clean_quality.windows) > 20
    assert clean_quality.windows[0].start == pytest.approx(0.0)
    assert clean_quality.windows[-1].end <= clean.duration + 1e-9


def test_evidence_explains_the_score(clean_quality):
    ev = clean_quality.evidence
    for key in (
        "baseline_stability", "qrs_visibility", "high_frequency_noise",
        "powerline_interference", "artifact_burden", "mean_penalties",
        "mean_features",
    ):
        assert key in ev
    # A clean record should have essentially nothing deducted.
    assert sum(ev["mean_penalties"].values()) < 10.0


@pytest.mark.parametrize("artifact", ARTIFACT_TYPES)
def test_corruption_lowers_the_local_score(clean, artifact):
    bad = corrupt_ecg(clean, artifact, start=8.0, duration=3.0, severity=0.9, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    inside = q.mean_score_between(8.5, 10.5)
    outside = q.mean_score_between(15.0, 25.0)
    assert inside < outside
    assert inside < QUALITY_TRUSTED_MIN


@pytest.mark.parametrize("artifact", ["baseline_wander", "powerline", "muscle_noise", "motion"])
def test_score_falls_as_severity_rises(clean, artifact):
    scores = []
    for sev in (0.3, 0.6, 0.95):
        bad = corrupt_ecg(clean, artifact, start=8.0, duration=3.0, severity=sev, seed=1)
        q = assess_quality(bad.signal, bad.sampling_rate)
        scores.append(q.mean_score_between(8.5, 10.5))
    assert scores[0] > scores[-1], f"{artifact}: {scores}"


def test_slow_heart_rate_is_not_mistaken_for_dropout():
    """A 1 s window can fall entirely between beats at 48 BPM."""
    rec = generate_ecg(heart_rate=45, seed=3)
    q = assess_quality(rec.signal, rec.sampling_rate)
    assert q.score > 90
    assert float(np.max([w.features["dropout_index"] for w in q.windows])) < 0.25


def test_score_window_is_pure_arithmetic():
    """Same features in, same score out -- nothing random anywhere."""
    feats = {
        "amplitude_ratio": 1.0, "baseline_drift_ratio": 0.01,
        "hf_noise_ratio": 0.01, "powerline_ratio": 0.001,
        "dropout_index": 0.0, "qrs_visibility": QRS_VISIBILITY_GOOD + 5,
    }
    a, _ = score_window(feats)
    b, _ = score_window(dict(feats))
    assert a == b == pytest.approx(100.0)


def test_energy_dropout_is_monotonic():
    values = [energy_dropout(v) for v in (1.0, 0.6, 0.4, 0.2, 0.0)]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_qrs_events_leave_no_gaps_on_a_clean_record(clean):
    """The property that matters for dropout: a clean record never goes quiet.

    ``qrs_event_times`` is not a beat detector and is not asserted to match the
    beat count exactly -- only that a healthy record keeps producing QRS-band
    energy at roughly the beat interval.
    """
    from src.quality import _bandpass_envelope, dropout_gap_threshold

    env = _bandpass_envelope(clean.signal, clean.sampling_rate)
    _, ref_env = record_references(clean.signal, env, clean.sampling_rate)
    events = qrs_event_times(env, clean.sampling_rate, ref_env)

    n_true = clean.metadata["true_beat_count"]
    assert n_true - 3 <= events.size <= n_true + 3
    gaps = np.diff(events)
    assert float(np.max(gaps)) < dropout_gap_threshold(events)


def test_too_short_a_signal_is_rejected():
    with pytest.raises(ValueError):
        assess_quality(np.zeros(50), 250)


# -------------------------------------------------------------- detection ---
def test_no_false_alarm_on_a_clean_record(clean_quality, clean):
    assert detect_artifacts(clean_quality, duration=clean.duration) == []


@pytest.mark.parametrize("artifact", ARTIFACT_TYPES)
def test_artifact_is_localised_and_classified(clean, artifact):
    bad = corrupt_ecg(clean, artifact, start=8.0, duration=3.0, severity=0.8, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    dets = detect_artifacts(q, duration=bad.duration)
    assert dets, f"{artifact} was not detected at all"
    best = max(
        dets,
        key=lambda d: max(0.0, min(11.0, d.end) - max(8.0, d.start)),
    )
    overlap = max(0.0, min(11.0, best.end) - max(8.0, best.start))
    assert overlap > 1.5, f"{artifact} localised to {best.start}-{best.end}"
    assert best.type == artifact
    assert 0.0 < best.confidence <= 0.98
    assert best.evidence


def test_detection_reports_its_supporting_features(clean):
    bad = corrupt_ecg(clean, "powerline", start=8.0, duration=3.0, severity=0.8, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    det = detect_artifacts(q, duration=bad.duration)[0]
    assert det.class_scores["powerline"] == max(det.class_scores.values())
    assert any("Hz" in e for e in det.evidence)


# --------------------------------------------------- recovery/revalidation ---
@pytest.mark.parametrize("artifact", ["baseline_wander", "powerline", "muscle_noise"])
def test_separable_artifacts_recover_and_revalidate(clean, artifact):
    bad = corrupt_ecg(clean, artifact, start=8.0, duration=3.0, severity=0.8, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    dets = detect_artifacts(q, duration=bad.duration)
    rep = recover(bad.signal, bad.sampling_rate, dets, q)
    assert rep.status == "VALIDATED"
    assert rep.affected_after > rep.affected_before


def test_contact_loss_is_not_declared_recovered(clean):
    """Filtering removes the step but cannot recreate the missing QRS content."""
    bad = corrupt_ecg(
        clean, "electrode_contact", start=8.0, duration=3.0, severity=0.8, seed=1
    )
    q = assess_quality(bad.signal, bad.sampling_rate)
    dets = detect_artifacts(q, duration=bad.duration)
    rep = recover(bad.signal, bad.sampling_rate, dets, q)
    assert rep.status != "VALIDATED"
    assert rep.rejected_regions


def test_severe_motion_recovery_fails_revalidation(clean):
    bad = corrupt_ecg(clean, "motion", start=8.0, duration=3.0, severity=0.95, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    dets = detect_artifacts(q, duration=bad.duration)
    rep = recover(bad.signal, bad.sampling_rate, dets, q)
    assert rep.status == "FAILED"
    assert all(not r.recovered for r in rep.regions)
    assert all(r.reason for r in rep.regions)


def test_no_detection_means_no_filtering(clean, clean_quality):
    rep = recover(clean.signal, clean.sampling_rate, [], clean_quality)
    assert rep.status == "NOT_REQUIRED"
    assert np.array_equal(rep.signal, clean.signal)
    assert rep.methods == []


def test_revalidation_reports_both_tests(clean):
    bad = corrupt_ecg(clean, "motion", start=8.0, duration=3.0, severity=0.95, seed=1)
    q = assess_quality(bad.signal, bad.sampling_rate)
    dets = detect_artifacts(q, duration=bad.duration)
    rep = recover(bad.signal, bad.sampling_rate, dets, q)
    for region in rep.regions:
        # The reason must mention both the score test and the structural test.
        assert "quality" in region.reason
        assert ";" in region.reason


# ----------------------------------------------------------------- filters ---
def test_highpass_removes_a_slow_drift(clean):
    t = clean.time
    drift = 0.8 * np.sin(2 * np.pi * 0.12 * t)
    out = highpass(clean.signal + drift, clean.sampling_rate, cutoff=0.5)
    assert np.std(out - clean.signal) < np.std(drift) * 0.3


def test_notch_removes_the_mains_tone(clean):
    t = clean.time
    tone = 0.3 * np.sin(2 * np.pi * 50.0 * t)
    out = notch(clean.signal + tone, clean.sampling_rate, 50.0)
    assert np.std(out - clean.signal) < np.std(tone) * 0.4


def test_lowpass_removes_high_frequency_noise(clean):
    rng = np.random.default_rng(0)
    noise = 0.25 * rng.standard_normal(clean.signal.size)
    out = lowpass(clean.signal + noise, clean.sampling_rate, cutoff=35.0)
    assert np.std(out - clean.signal) < np.std(noise) * 0.7
