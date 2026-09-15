"""Tests for the synthetic ECG generator."""

import numpy as np
import pytest

from src.ecg_generator import (
    TRUE_QRS_DURATION_MS,
    ECGRecord,
    generate_ecg,
    noise_free_twin,
)


def test_shape_and_duration():
    rec = generate_ecg(duration=10, sampling_rate=250)
    assert rec.signal.size == 2500
    assert rec.time.size == 2500
    assert rec.duration == pytest.approx(10.0)
    assert isinstance(rec, ECGRecord)


def test_is_deterministic_for_a_seed():
    a = generate_ecg(seed=42)
    b = generate_ecg(seed=42)
    assert np.array_equal(a.signal, b.signal)
    assert np.array_equal(a.r_peaks_true, b.r_peaks_true)


def test_different_seeds_give_different_signals():
    a = generate_ecg(seed=1)
    b = generate_ecg(seed=2)
    assert not np.array_equal(a.signal, b.signal)


@pytest.mark.parametrize("hr", [45, 60, 72, 100, 140])
def test_beat_count_matches_requested_rate(hr):
    rec = generate_ecg(duration=30, heart_rate=hr, seed=3)
    expected = 30.0 * hr / 60.0
    # The first beat starts at 0.35 s, so one fewer beat than the ideal is normal.
    assert abs(rec.metadata["true_beat_count"] - expected) <= max(2.0, 0.1 * expected)
    assert rec.metadata["true_mean_hr_bpm"] == pytest.approx(hr, rel=0.06)


def test_rr_variability_is_mild_and_present():
    rec = generate_ecg(duration=60, seed=5)
    rr = np.diff(rec.r_peaks_true) / rec.sampling_rate
    cv = np.std(rr) / np.mean(rr)
    assert 0.0 < cv < 0.12, "RR variability should be present but physiological"


def test_waveform_has_plausible_amplitude():
    rec = generate_ecg(seed=7)
    assert 0.5 < rec.signal.max() < 2.0
    assert -0.6 < rec.signal.min() < 0.0


def test_r_peaks_sit_at_local_maxima():
    rec = generate_ecg(seed=11)
    fs = rec.sampling_rate
    for idx in rec.r_peaks_true[2:-2]:
        window = rec.signal[idx - fs // 10: idx + fs // 10]
        assert rec.signal[idx] >= window.max() - 0.12


def test_metadata_records_ground_truth():
    rec = generate_ecg(seed=13)
    for key in (
        "true_mean_hr_bpm", "true_beat_count", "true_qrs_duration_ms",
        "seed", "generator_kwargs", "artifacts",
    ):
        assert key in rec.metadata
    assert rec.metadata["true_qrs_duration_ms"] == TRUE_QRS_DURATION_MS
    assert rec.metadata["artifacts"] == []


def test_noise_free_twin_keeps_beats_and_drops_noise():
    rec = generate_ecg(seed=17)
    twin = noise_free_twin(rec)
    assert np.array_equal(rec.r_peaks_true, twin.r_peaks_true)
    assert np.max(np.abs(rec.signal - twin.signal)) > 0.0
    # Before the first P wave (the first R is at 0.35 s, its P wave at ~0.15 s)
    # the twin should carry no signal at all.
    quiet = twin.signal[: int(0.04 * twin.sampling_rate)]
    assert np.max(np.abs(quiet)) < 1e-6
    assert np.max(np.abs(rec.signal[: int(0.04 * rec.sampling_rate)])) > 0.0


@pytest.mark.parametrize(
    "kwargs", [{"duration": 0}, {"sampling_rate": 0}, {"heart_rate": 5}, {"heart_rate": 400}]
)
def test_rejects_impossible_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_ecg(**kwargs)


def test_copy_with_preserves_ground_truth():
    rec = generate_ecg(seed=19)
    other = rec.copy_with(np.zeros_like(rec.signal), note="x")
    assert np.array_equal(other.r_peaks_true, rec.r_peaks_true)
    assert other.metadata["note"] == "x"
    assert not np.array_equal(other.signal, rec.signal)
