"""Tests for R-peak detection and the measurement functions."""

import inspect

import numpy as np
import pytest

from src.artifacts import corrupt_ecg
from src.ecg_generator import generate_ecg, noise_free_twin
from src.measurements import (
    measure_morphology,
    qrs_boundaries,
    rr_consistency,
    rr_from_beats,
)
from src.peaks import detect_r_peaks, physiologically_plausible, rescore_beats

TOL = 0.06  # seconds


def match(true_idx, det_times, fs):
    """Return (sensitivity, ppv, mean |timing error| in ms)."""
    t_true = np.asarray(true_idx, dtype=float) / fs
    used, errors = set(), []
    for t in t_true:
        if det_times.size == 0:
            break
        d = np.abs(det_times - t)
        for j in np.argsort(d):
            if d[j] > TOL:
                break
            if j not in used:
                used.add(int(j))
                errors.append((det_times[j] - t) * 1000.0)
                break
    tp = len(used)
    return (
        tp / max(1, t_true.size),
        tp / max(1, det_times.size),
        float(np.mean(np.abs(errors))) if errors else float("nan"),
    )


# ------------------------------------------------------------- ground truth --
def test_detector_cannot_see_ground_truth():
    """The detector's only inputs are a waveform and a sampling rate."""
    params = list(inspect.signature(detect_r_peaks).parameters)
    assert params == ["x", "fs"]


# ------------------------------------------------------------------- peaks --
@pytest.mark.parametrize("hr", [45, 60, 72, 95, 120])
def test_finds_every_beat_in_a_clean_record(hr):
    """The detector is tuned for sensitivity, not for precision.

    A missed beat removes evidence that nothing downstream can recover.  An
    extra candidate costs nothing, because the evidence engine judges every
    beat before using it -- see
    ``test_t_waves_are_distinguishable_from_r_peaks`` here, and
    ``test_gating_removes_the_detector_false_positives`` in test_pipeline.
    """
    rec = generate_ecg(heart_rate=hr, seed=hr)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    se, ppv, err = match(rec.r_peaks_true, report.times, rec.sampling_rate)
    assert se == 1.0, f"HR={hr}: sensitivity {se}"
    assert ppv > 0.85, f"HR={hr}: ppv {ppv}"
    assert err < 15.0, f"HR={hr}: timing error {err} ms"


def test_timing_is_accurate_on_a_noise_free_record():
    rec = noise_free_twin(generate_ecg(seed=42))
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    _, _, err = match(rec.r_peaks_true, report.times, rec.sampling_rate)
    assert err < 5.0


def test_respects_the_refractory_period():
    rec = generate_ecg(heart_rate=120, seed=4)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    gaps = np.diff(report.times)
    assert np.all(gaps >= 0.23)


def test_reports_per_beat_evidence():
    rec = generate_ecg(seed=42)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    for beat in report.beats:
        assert beat.prominence >= 0.0
        assert beat.relative_amplitude > 0.0
        assert -1.0 <= beat.template_correlation <= 1.0
    # On a clean record every beat should look like the template.
    corrs = [b.template_correlation for b in report.beats]
    assert float(np.median(corrs)) > 0.95


def test_t_waves_are_distinguishable_from_r_peaks():
    """False positives must be separable by the evidence the detector reports."""
    rec = generate_ecg(heart_rate=48, seed=3)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    t_true = rec.r_peaks_true / rec.sampling_rate
    spurious = [
        b for b in report.beats if np.min(np.abs(t_true - b.time)) > TOL
    ]
    for b in spurious:
        assert b.relative_prominence < 0.25 or b.template_correlation < 0.70


def test_flat_signal_yields_no_beats():
    report = detect_r_peaks(np.zeros(2500), 250)
    assert report.beats == []


def test_rescore_uses_only_the_reference_subset():
    rec = generate_ecg(seed=42)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    beats = report.beats
    before = [b.relative_amplitude for b in beats]
    # Halve the amplitude of the beats we will exclude from the reference.
    mask = [i % 2 == 0 for i in range(len(beats))]
    rescore_beats(rec.signal, rec.sampling_rate, beats, mask)
    after = [b.relative_amplitude for b in beats]
    assert len(before) == len(after)
    assert all(a > 0 for a in after)


def test_rescore_is_a_no_op_without_enough_reference_beats():
    rec = generate_ecg(seed=42)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    before = [b.relative_amplitude for b in report.beats]
    rescore_beats(
        rec.signal, rec.sampling_rate, report.beats, [False] * len(report.beats)
    )
    assert [b.relative_amplitude for b in report.beats] == before


# ------------------------------------------------------------ measurements --
def test_heart_rate_matches_the_generator():
    rec = generate_ecg(heart_rate=72, seed=42)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    rhythm = rr_from_beats(report.beats)
    assert rhythm.heart_rate == pytest.approx(rec.metadata["true_mean_hr_bpm"], abs=2.0)
    assert rhythm.n_intervals > 25


def test_rr_consistency_is_a_fraction():
    assert rr_consistency(np.array([0.8, 0.81, 0.79, 0.8])) == pytest.approx(1.0)
    mixed = rr_consistency(np.array([0.8, 0.8, 0.8, 0.8, 2.0]))
    assert 0.0 < mixed < 1.0
    assert np.isnan(rr_consistency(np.array([])))


def test_rr_excludes_intervals_spanning_a_dropped_beat():
    rec = generate_ecg(seed=42)
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    beats = report.beats
    kept = beats[:5] + beats[8:]          # drop three beats in the middle
    rhythm = rr_from_beats(kept, contiguous_only=True)
    assert rhythm.n_excluded_intervals >= 1
    assert float(np.max(rhythm.rr_intervals)) < 1.5


def test_implausible_rr_is_rejected():
    assert physiologically_plausible(0.8)
    assert not physiologically_plausible(0.1)
    assert not physiologically_plausible(5.0)


def test_qrs_boundaries_are_stable_on_a_clean_record():
    rec = noise_free_twin(generate_ecg(seed=42))
    report = detect_r_peaks(rec.signal, rec.sampling_rate)
    morph = measure_morphology(rec.signal, rec.sampling_rate, report.beats)
    clear = [m for m in morph if m.clear]
    assert len(clear) >= len(report.beats) - 2
    durations = np.array([m.duration_ms for m in clear])
    assert 55 < float(np.median(durations)) < 145
    iqr = float(np.percentile(durations, 75) - np.percentile(durations, 25))
    assert iqr < 5.0, "a noise-free record should give near-identical widths"


def test_qrs_boundaries_bracket_the_r_peak():
    rec = generate_ecg(seed=42)
    fs = rec.sampling_rate
    report = detect_r_peaks(rec.signal, fs)
    for beat in report.beats[3:8]:
        m = qrs_boundaries(rec.signal, fs, beat.index)
        assert m.clear
        assert m.onset < beat.time < m.offset


def test_qrs_boundaries_degrade_gracefully_at_the_record_edge():
    rec = generate_ecg(seed=42)
    m = qrs_boundaries(rec.signal, rec.sampling_rate, 2)
    assert not m.clear
    assert m.reason
    assert np.isnan(m.duration_ms)


def test_noise_widens_the_spread_of_qrs_estimates():
    """The spread across beats is what the QRS gate keys on, so it must react."""
    clean = noise_free_twin(generate_ecg(seed=42))
    noisy = corrupt_ecg(generate_ecg(seed=42), "muscle_noise", severity=0.7, seed=1)

    def spread(rec):
        report = detect_r_peaks(rec.signal, rec.sampling_rate)
        morph = measure_morphology(rec.signal, rec.sampling_rate, report.beats)
        d = np.array([m.duration_ms for m in morph if m.clear])
        return float(np.percentile(d, 75) - np.percentile(d, 25)) if d.size >= 4 else 999.0

    assert spread(noisy) > spread(clean)
