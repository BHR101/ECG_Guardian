"""End-to-end tests: the pipeline, the evidence engine and the demo scenarios.

The tests that matter most here are the ones asserting the project's actual
claim -- that two measurements taken from one record can receive different
verdicts, and that a failed recovery is never treated as a successful one.
"""

import numpy as np
import pytest

from src.artifacts import corrupt_ecg
from src.config import (
    TRUST_DEGRADED,
    TRUST_RECOVERED,
    TRUST_REJECTED,
    TRUST_TRUSTED,
)
from src.demo import SCENARIOS, SCENARIOS_BY_KEY, build
from src.ecg_generator import generate_ecg
from src.evidence import ACCEPTED, REJECTED
from src.pipeline import run_pipeline

MEASUREMENTS = ("heart_rate", "rr_interval", "rr_consistency", "qrs_duration")


@pytest.fixture(scope="module")
def clean_result():
    return run_pipeline(generate_ecg(seed=42), "clean")


@pytest.fixture(scope="module")
def results():
    return {s.key: run_pipeline(s.record(), s.name) for s in SCENARIOS}


# --------------------------------------------------------------- structure --
def test_pipeline_produces_every_stage(clean_result):
    assert clean_result.quality_raw.windows
    assert clean_result.quality_processed is not None
    assert clean_result.beats
    assert len(clean_result.morphology) == len(clean_result.beats)
    assert clean_result.trust_map
    assert clean_result.log
    assert len(clean_result.measurements) == len(MEASUREMENTS)
    assert {m.measurement for m in clean_result.measurements} == set(MEASUREMENTS)
    assert clean_result.errors == []


def test_trust_map_tiles_the_record_without_gaps(results):
    for key, result in results.items():
        segs = result.trust_map
        assert segs[0].start == pytest.approx(0.0), key
        assert segs[-1].end == pytest.approx(result.record.duration, abs=0.6), key
        for a, b in zip(segs[:-1], segs[1:]):
            assert b.start == pytest.approx(a.end), f"{key}: gap or overlap at {a.end}"
        for s in segs:
            assert s.label in (
                TRUST_TRUSTED, TRUST_DEGRADED, TRUST_RECOVERED, TRUST_REJECTED
            )


def test_every_log_entry_is_timestamped(clean_result):
    for entry in clean_result.log:
        assert entry.clock and entry.stage and entry.message
        assert entry.level in ("info", "warn", "error")


def test_summary_is_serialisable(clean_result):
    import json

    text = json.dumps(clean_result.summary(), default=str)
    assert "measurements" in text and "trust_map" in text


# ------------------------------------------------------------ clean record --
def test_clean_record_is_fully_trusted(clean_result):
    assert clean_result.quality_raw.score > 90
    assert clean_result.detections == []
    assert clean_result.recovery.status == "NOT_REQUIRED"
    assert clean_result.overall_trust == TRUST_TRUSTED
    assert all(s.label == TRUST_TRUSTED for s in clean_result.trust_map)


def test_clean_record_reports_every_measurement(clean_result):
    for m in clean_result.measurements:
        assert m.status == ACCEPTED, f"{m.measurement}: {m.reason}"
        assert m.value is not None
        assert m.confidence > 0.7


def test_heart_rate_is_accurate_on_a_clean_record(clean_result):
    hr = clean_result.measurement("heart_rate")
    true_hr = clean_result.record.metadata["true_mean_hr_bpm"]
    assert abs(hr.value - true_hr) < 1.5


def test_gating_removes_the_detector_false_positives():
    """Extra candidates from the detector must not survive into a measurement."""
    rec = generate_ecg(heart_rate=45, seed=3)
    result = run_pipeline(rec, "slow")
    hr = result.measurement("heart_rate")
    t_true = rec.r_peaks_true / rec.sampling_rate

    used = [
        b for b, v in zip(result.beats, hr.beat_verdicts) if v.accepted
    ]
    wrong = [b for b in used if np.min(np.abs(t_true - b.time)) > 0.06]
    assert not wrong, f"{len(wrong)} spurious beats were used for heart rate"
    assert hr.status == ACCEPTED


# -------------------------------------------- the central claim of the work --
def test_one_record_can_produce_different_verdicts():
    """Measurement-specific gating: heart rate accepted, QRS width refused.

    A single global quality gate cannot produce this outcome, so this is the
    test that most directly checks the project's claim.
    """
    result = run_pipeline(build("muscle_noise"), "muscle_noise")
    hr = result.measurement("heart_rate")
    qrs = result.measurement("qrs_duration")
    assert hr.status == ACCEPTED
    assert qrs.status == REJECTED
    assert qrs.value is None
    assert qrs.reason


def test_the_two_gates_consume_different_evidence():
    """The same beats are judged against different standards."""
    result = run_pipeline(build("muscle_noise"), "muscle_noise")
    hr = result.measurement("heart_rate")
    qrs = result.measurement("qrs_duration")
    assert hr.evidence_standard != qrs.evidence_standard
    hr_ok = {v.time for v in hr.beat_verdicts if v.accepted}
    qrs_ok = {v.time for v in qrs.beat_verdicts if v.accepted}
    assert qrs_ok <= hr_ok, "the morphology standard must be the stricter one"
    assert len(qrs_ok) < len(hr_ok)


def test_rejected_measurement_reports_no_number():
    result = run_pipeline(build("severe"), "severe")
    for m in result.measurements:
        if m.status == REJECTED:
            assert m.value is None
            assert "NOT REPORTED" in m.display_value
            assert m.reason


def test_severe_corruption_refuses_everything():
    result = run_pipeline(build("severe"), "severe")
    assert result.overall_trust == TRUST_REJECTED
    assert result.accepted == []
    assert len(result.rejected) == len(MEASUREMENTS)


# ------------------------------------------------- recovery / revalidation --
def test_failed_recovery_rejects_the_segment():
    rec = corrupt_ecg(
        generate_ecg(seed=42), "motion", start=8.0, duration=3.0, severity=0.95, seed=1
    )
    result = run_pipeline(rec, "motion")
    assert result.recovery.status == "FAILED"
    rejected = [s for s in result.trust_map if s.label == TRUST_REJECTED]
    assert rejected, "a failed recovery must leave a rejected segment"
    covered = any(s.start <= 9.5 <= s.end for s in rejected)
    assert covered, "the rejected segment must cover the artifact"


def test_beats_inside_a_failed_region_are_never_used():
    rec = corrupt_ecg(
        generate_ecg(seed=42), "motion", start=8.0, duration=3.0, severity=0.95, seed=1
    )
    result = run_pipeline(rec, "motion")
    rejected = result.recovery.rejected_regions
    assert rejected
    hr = result.measurement("heart_rate")
    for beat, verdict in zip(result.beats, hr.beat_verdicts):
        if any(s <= beat.time < e for s, e in rejected):
            assert not verdict.accepted, f"beat at {beat.time} was inside a failed region"


def test_validated_recovery_is_marked_recovered():
    result = run_pipeline(build("motion_recoverable"), "motion_recoverable")
    assert result.recovery.status == "VALIDATED"
    assert any(s.label == TRUST_RECOVERED for s in result.trust_map)
    assert result.recovery.affected_after > result.recovery.affected_before


def test_recovery_is_always_revalidated(results):
    """No region may be called recovered without a recorded re-check."""
    for key, result in results.items():
        for region in result.recovery.regions:
            assert region.reason, key
            assert region.score_after == pytest.approx(
                result.quality_processed.mean_score_between(region.start, region.end),
                abs=1e-6,
            ), key


# ------------------------------------------------------------- confidence --
def test_confidence_is_the_sum_of_its_measured_terms(results):
    for key, result in results.items():
        for m in result.measurements:
            if not m.confidence_terms:
                continue
            total = sum(m.confidence_terms.values())
            assert m.confidence == pytest.approx(min(total, 0.99), abs=0.01), (
                f"{key}/{m.measurement}: {m.confidence} vs {total}"
            )


def test_confidence_is_bounded(results):
    for result in results.values():
        for m in result.measurements:
            assert 0.0 <= m.confidence <= 0.99


def test_every_measurement_carries_its_criteria(results):
    for key, result in results.items():
        for m in result.measurements:
            assert m.criteria, f"{key}/{m.measurement}"
            assert any(c.name == "measurement confidence" for c in m.criteria)
            if m.status == ACCEPTED:
                assert all(c.passed for c in m.criteria), f"{key}/{m.measurement}"
            else:
                assert any(not c.passed for c in m.criteria), f"{key}/{m.measurement}"


# ---------------------------------------------------------------- scenarios --
def test_scenarios_are_deterministic():
    for scenario in SCENARIOS:
        a = run_pipeline(scenario.record(), scenario.name)
        b = run_pipeline(scenario.record(), scenario.name)
        assert np.array_equal(a.record.signal, b.record.signal), scenario.key
        assert [m.value for m in a.measurements] == [m.value for m in b.measurements]
        assert [(d.type, d.start, d.end) for d in a.detections] == [
            (d.type, d.start, d.end) for d in b.detections
        ]


def test_every_scenario_runs_without_error(results):
    for key, result in results.items():
        assert result.errors == [], f"{key}: {result.errors}"
        assert result.trust_map
        assert len(result.measurements) == len(MEASUREMENTS)


def test_scenario_keys_are_unique():
    keys = [s.key for s in SCENARIOS]
    assert len(keys) == len(set(keys)) == len(SCENARIOS_BY_KEY)


def test_artifact_scenarios_localise_their_artifact(results):
    for key in ("motion_recoverable", "motion_unrecoverable"):
        result = results[key]
        assert result.detections, key
        det = result.detections[0]
        assert det.type == "motion"
        overlap = max(0.0, min(11.0, det.end) - max(8.0, det.start))
        assert overlap > 2.0, f"{key}: localised to {det.start}-{det.end}"


# ------------------------------------------------------------- robustness --
def test_pipeline_survives_a_degenerate_signal():
    """A flat trace must produce refusals, not an exception."""
    rec = generate_ecg(seed=42)
    flat = rec.copy_with(np.zeros_like(rec.signal))
    result = run_pipeline(flat, "flat")
    assert result.measurements
    assert all(m.status == REJECTED for m in result.measurements)


def test_pipeline_survives_a_constant_offset():
    rec = generate_ecg(seed=42)
    odd = rec.copy_with(np.full_like(rec.signal, 3.0))
    result = run_pipeline(odd, "constant")
    assert all(m.status == REJECTED for m in result.measurements)


def test_pipeline_handles_a_short_record():
    rec = generate_ecg(duration=6, seed=42)
    result = run_pipeline(rec, "short")
    assert result.trust_map
    assert len(result.measurements) == len(MEASUREMENTS)
