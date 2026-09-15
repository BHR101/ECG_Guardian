"""Tests for controlled artifact injection."""

import numpy as np
import pytest

from src.artifacts import ARTIFACT_TYPES, corrupt_ecg, corrupt_many
from src.config import POWERLINE_FREQ
from src.ecg_generator import generate_ecg


@pytest.fixture
def base():
    return generate_ecg(seed=42)


@pytest.mark.parametrize("artifact", ARTIFACT_TYPES)
def test_injection_is_localised(base, artifact):
    """Nothing outside the requested span may change."""
    out = corrupt_ecg(base, artifact, start=8.0, duration=3.0, severity=0.8, seed=1)
    fs = base.sampling_rate
    delta = out.signal - base.signal
    outside = np.concatenate([delta[: 8 * fs], delta[11 * fs:]])
    assert np.max(np.abs(outside)) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(delta[8 * fs: 11 * fs])) > 0.0


@pytest.mark.parametrize("artifact", ARTIFACT_TYPES)
def test_metadata_records_ground_truth(base, artifact):
    out = corrupt_ecg(base, artifact, start=8.0, duration=3.0, severity=0.6, seed=1)
    spec = out.metadata["artifacts"][-1]
    assert spec["type"] == artifact
    assert spec["start"] == pytest.approx(8.0)
    assert spec["end"] == pytest.approx(11.0)
    assert spec["severity"] == pytest.approx(0.6)
    assert isinstance(spec["recoverable"], bool)
    assert spec["notes"]


@pytest.mark.parametrize("artifact", ARTIFACT_TYPES)
def test_severity_scales_the_disturbance(base, artifact):
    fs = base.sampling_rate
    mild = corrupt_ecg(base, artifact, start=8.0, duration=3.0, severity=0.3, seed=1)
    harsh = corrupt_ecg(base, artifact, start=8.0, duration=3.0, severity=0.9, seed=1)
    d_mild = np.std((mild.signal - base.signal)[8 * fs: 11 * fs])
    d_harsh = np.std((harsh.signal - base.signal)[8 * fs: 11 * fs])
    assert d_harsh > d_mild


def test_source_record_is_never_modified(base):
    before = base.signal.copy()
    corrupt_ecg(base, "motion", start=5.0, duration=2.0, severity=0.9, seed=1)
    assert np.array_equal(base.signal, before)
    assert base.metadata["artifacts"] == []


def test_powerline_lands_at_the_mains_frequency(base):
    out = corrupt_ecg(base, "powerline", severity=0.9, seed=1)
    fs = base.sampling_rate
    delta = out.signal - base.signal
    spectrum = np.abs(np.fft.rfft(delta))
    freqs = np.fft.rfftfreq(delta.size, 1.0 / fs)
    peak = freqs[int(np.argmax(spectrum))]
    assert peak == pytest.approx(POWERLINE_FREQ, abs=1.0)


def test_baseline_wander_is_low_frequency(base):
    out = corrupt_ecg(base, "baseline_wander", severity=0.9, seed=1)
    delta = out.signal - base.signal
    freqs = np.fft.rfftfreq(delta.size, 1.0 / base.sampling_rate)
    spectrum = np.abs(np.fft.rfft(delta))
    # Almost all of the injected energy must sit below 1 Hz.
    below = spectrum[freqs < 1.0].sum()
    assert below / spectrum.sum() > 0.9


def test_muscle_noise_is_high_frequency(base):
    out = corrupt_ecg(base, "muscle_noise", severity=0.9, seed=1)
    delta = out.signal - base.signal
    freqs = np.fft.rfftfreq(delta.size, 1.0 / base.sampling_rate)
    spectrum = np.abs(np.fft.rfft(delta))
    above = spectrum[freqs > 20.0].sum()
    assert above / spectrum.sum() > 0.8


def test_electrode_contact_removes_qrs_amplitude(base):
    fs = base.sampling_rate
    out = corrupt_ecg(
        base, "electrode_contact", start=8.0, duration=3.0, severity=0.95, seed=1
    )
    # Take a span that is inside the region, past the taper ramp and past the
    # step discontinuity, so what is left is the flat trace itself.
    mid = slice(int(9.3 * fs), int(10.2 * fs))
    assert np.ptp(out.signal[mid]) < 0.4 * np.ptp(base.signal[mid])


def test_unknown_artifact_type_is_rejected(base):
    with pytest.raises(ValueError):
        corrupt_ecg(base, "not_a_real_artifact")


def test_severity_is_clamped(base):
    out = corrupt_ecg(base, "motion", start=2.0, duration=1.0, severity=5.0, seed=1)
    assert out.metadata["artifacts"][-1]["severity"] == pytest.approx(1.0)


def test_corrupt_many_stacks_artifacts(base):
    out = corrupt_many(
        base,
        [
            {"artifact_type": "baseline_wander", "start": 2.0, "duration": 3.0},
            {"artifact_type": "motion", "start": 10.0, "duration": 2.0},
        ],
    )
    assert len(out.metadata["artifacts"]) == 2
    assert {a["type"] for a in out.metadata["artifacts"]} == {"baseline_wander", "motion"}


def test_injection_is_deterministic(base):
    a = corrupt_ecg(base, "muscle_noise", start=4.0, duration=3.0, severity=0.7, seed=5)
    b = corrupt_ecg(base, "muscle_noise", start=4.0, duration=3.0, severity=0.7, seed=5)
    assert np.array_equal(a.signal, b.signal)
