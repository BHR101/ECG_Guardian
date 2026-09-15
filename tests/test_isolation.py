"""Phase 7 -- ground-truth isolation audit.

Signature inspection shows that no analysis entry point *accepts* ground truth.
That is necessary but not sufficient: the record object carries its own truth
in ``r_peaks_true`` and ``metadata``, and any stage could reach into it.

So this file proves isolation by experiment as well as by inspection.  The
same waveform is analysed twice -- once with correct ground truth attached and
once with the ground truth destroyed -- and every decision the pipeline makes
must come out bit-for-bit identical.  If any stage were peeking, the two runs
would diverge.
"""

import inspect

import numpy as np
import pytest

from src.artifacts import corrupt_ecg
from src.demo import SCENARIOS
from src.ecg_generator import ECGRecord, generate_ecg
from src.pipeline import run_pipeline

ANALYSIS_MODULES = [
    "quality", "detection", "filtering", "peaks", "measurements",
    "evidence", "pipeline", "plots",
]

TRUTH_FIELDS = ("r_peaks_true", "true_mean_hr_bpm", "true_beat_count",
                "true_qrs_duration_ms", "generator_kwargs")


def _blinded(record: ECGRecord) -> ECGRecord:
    """The same waveform with its ground truth destroyed, not merely removed.

    Deliberately filled with *wrong* values rather than empty ones: a stage
    that read them would then produce a different answer, whereas absent
    values might simply be skipped.
    """
    rng = np.random.default_rng(0)
    fake_peaks = np.sort(
        rng.choice(record.signal.size, size=max(2, record.signal.size // 700),
                   replace=False)
    )
    return ECGRecord(
        time=record.time,
        signal=record.signal,
        sampling_rate=record.sampling_rate,
        r_peaks_true=fake_peaks,
        metadata={
            "source": "blinded",
            "true_mean_hr_bpm": 999.0,
            "true_beat_count": -1,
            "true_qrs_duration_ms": -1.0,
            "artifacts": [{"type": "nonsense", "start": 0.0, "end": 1.0,
                           "severity": 1.0, "recoverable": False, "notes": ""}],
        },
    )


def _fingerprint(result) -> dict:
    """Everything the pipeline decided, in a directly comparable form."""
    return {
        "quality_raw": round(result.quality_raw.score, 6),
        "quality_processed": round(result.quality_processed.score, 6),
        "window_scores": [round(w.score, 6) for w in result.quality_raw.windows],
        "detections": [
            (d.type, d.start, d.end, d.confidence) for d in result.detections
        ],
        "recovery": (
            result.recovery.status,
            [(r.start, r.end, r.recovered, r.reason) for r in result.recovery.regions],
        ),
        "beats": [round(b.time, 6) for b in result.beats],
        "beat_evidence": [
            (round(b.relative_amplitude, 6), round(b.relative_prominence, 6),
             round(b.template_correlation, 6))
            for b in result.beats
        ],
        "trust_map": [(s.label, round(s.start, 6), round(s.end, 6))
                      for s in result.trust_map],
        "measurements": [
            (m.measurement, m.status, m.value, m.confidence, m.reason,
             m.validated_beats, m.rejected_beats)
            for m in result.measurements
        ],
    }


CASES = [
    ("clean", lambda: generate_ecg(seed=42)),
    ("slow", lambda: generate_ecg(heart_rate=42, seed=7)),
    ("fast", lambda: generate_ecg(heart_rate=140, seed=8)),
    ("motion", lambda: corrupt_ecg(generate_ecg(seed=42), "motion",
                                   start=8.0, duration=3.0, severity=0.8, seed=1)),
    ("contact", lambda: corrupt_ecg(generate_ecg(seed=42), "electrode_contact",
                                    start=8.0, duration=4.0, severity=0.95, seed=1)),
    ("muscle", lambda: corrupt_ecg(generate_ecg(seed=42), "muscle_noise",
                                   severity=0.7, seed=1)),
]


@pytest.mark.parametrize("name,build", CASES, ids=[c[0] for c in CASES])
def test_analysis_is_identical_with_ground_truth_destroyed(name, build):
    """The experiment: destroy the truth, and nothing about the analysis moves."""
    record = build()
    truthful = _fingerprint(run_pipeline(record, name))
    blinded = _fingerprint(run_pipeline(_blinded(record), name))
    assert truthful == blinded, f"{name}: analysis changed when ground truth changed"


def test_every_demo_scenario_is_blind_to_its_ground_truth():
    for scenario in SCENARIOS:
        record = scenario.record()
        a = _fingerprint(run_pipeline(record, scenario.key))
        b = _fingerprint(run_pipeline(_blinded(record), scenario.key))
        assert a == b, f"{scenario.key}: analysis depends on ground truth"


def test_no_analysis_module_mentions_a_ground_truth_field():
    """Static audit: the words do not even appear in the analysis path."""
    import importlib
    import pathlib

    offenders = {}
    for name in ANALYSIS_MODULES:
        module = importlib.import_module(f"src.{name}")
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        hits = [f for f in TRUTH_FIELDS if f in source]
        if hits:
            offenders[name] = hits
    assert not offenders, f"ground-truth fields referenced in: {offenders}"


def test_no_analysis_entry_point_accepts_ground_truth():
    from src import detection, evidence, filtering, peaks, quality

    entry_points = [
        quality.assess_quality,
        detection.detect_artifacts,
        filtering.recover,
        peaks.detect_r_peaks,
        evidence.run_evidence_engine,
    ]
    banned = ("truth", "r_peaks_true", "record", "expected", "label")
    for fn in entry_points:
        params = list(inspect.signature(fn).parameters)
        bad = [p for p in params if any(b in p for b in banned)]
        assert not bad, f"{fn.__name__} accepts {bad}"


def test_pipeline_touches_only_waveform_fields_of_the_record():
    """`run_pipeline` may read the signal, the rate and the duration -- no more."""
    import pathlib
    import re

    from src import pipeline

    source = pathlib.Path(pipeline.__file__).read_text(encoding="utf-8")
    body = source[source.index("def run_pipeline"):]
    allowed = {"signal", "duration", "sampling_rate"}
    used = set(re.findall(r"record\.([A-Za-z_][A-Za-z0-9_]*)", body))
    assert used <= allowed, (
        f"run_pipeline reads unexpected record fields: {sorted(used - allowed)}"
    )
