"""Phase 13 -- deterministic demo scenarios.

Every scenario is built from a fixed seed and fixed artifact parameters, so the
same run produces the same waveform, the same detections and the same verdicts
every time.  Nothing about the presentation depends on chance.

The ``expectation`` text on each scenario says what the scenario is *for*.  It
is a description of the situation, not a stored result: the numbers shown in
the dashboard always come from running the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .artifacts import corrupt_ecg, corrupt_many
from .ecg_generator import ECGRecord, generate_ecg

DEMO_SEED = 42


@dataclass
class Scenario:
    """One reproducible demo case."""

    key: str
    name: str
    description: str
    expectation: str
    build: Callable[[], ECGRecord]

    def record(self) -> ECGRecord:
        return self.build()


def _clean() -> ECGRecord:
    return generate_ecg(seed=DEMO_SEED)


def _baseline_wander() -> ECGRecord:
    return corrupt_ecg(
        generate_ecg(seed=DEMO_SEED),
        "baseline_wander",
        start=5.0,
        duration=12.0,
        severity=0.85,
        seed=11,
    )


def _powerline() -> ECGRecord:
    return corrupt_ecg(
        generate_ecg(seed=DEMO_SEED),
        "powerline",
        severity=0.8,
        seed=12,
    )


def _muscle_noise() -> ECGRecord:
    # Applied to the whole record on purpose.  Every beat stays reliably
    # *timed* while every beat's morphology is degraded -- which is exactly
    # the situation measurement-specific gating exists to handle.
    return corrupt_ecg(
        generate_ecg(seed=DEMO_SEED),
        "muscle_noise",
        severity=0.6,
        seed=13,
    )


def _motion_recoverable() -> ECGRecord:
    return corrupt_ecg(
        generate_ecg(seed=DEMO_SEED),
        "motion",
        start=8.0,
        duration=3.0,
        severity=0.4,
        seed=14,
    )


def _motion_unrecoverable() -> ECGRecord:
    return corrupt_ecg(
        generate_ecg(seed=DEMO_SEED),
        "motion",
        start=8.0,
        duration=3.0,
        severity=0.9,
        seed=15,
    )


def _severe() -> ECGRecord:
    # Contact loss across most of the record plus a violent motion burst in
    # what is left.  There is genuinely not enough signal here to measure
    # anything, and the system is expected to say so.
    return corrupt_many(
        generate_ecg(seed=DEMO_SEED),
        [
            {
                "artifact_type": "electrode_contact",
                "start": 3.0,
                "duration": 21.0,
                "severity": 0.95,
            },
            {
                "artifact_type": "motion",
                "start": 25.0,
                "duration": 4.0,
                "severity": 0.95,
            },
        ],
    )


SCENARIOS: list[Scenario] = [
    Scenario(
        key="clean",
        name="Clean ECG",
        description="Synthetic 30 s record at ~72 BPM with only the sensor noise floor.",
        expectation=(
            "High signal quality, no artifact region, every measurement should "
            "have enough evidence to be reported."
        ),
        build=_clean,
    ),
    Scenario(
        key="baseline_wander",
        name="Baseline wander",
        description="Low-frequency drift injected over 5-17 s.",
        expectation=(
            "Drift localised and classified; baseline correction should recover "
            "the region, and revalidation should confirm the drift is gone."
        ),
        build=_baseline_wander,
    ),
    Scenario(
        key="powerline",
        name="Power-line interference",
        description="50 Hz mains interference across the whole record.",
        expectation=(
            "Narrowband interference identified and removed by a notch filter; "
            "recovery should revalidate."
        ),
        build=_powerline,
    ),
    Scenario(
        key="muscle_noise",
        name="Muscle noise (mild degradation)",
        description="EMG-like broadband noise across the whole record.",
        expectation=(
            "The key case for measurement-specific gating: R-peak timing should "
            "survive, so heart rate is reportable, while QRS onset and offset "
            "should not be resolvable consistently enough to report a width."
        ),
        build=_muscle_noise,
    ),
    Scenario(
        key="motion_recoverable",
        name="Motion artifact (recoverable)",
        description="Localised motion disturbance over 8-11 s, moderate severity.",
        expectation=(
            "Artifact localised to its actual span, conservative recovery "
            "attempted, and revalidation should confirm the region is usable "
            "again."
        ),
        build=_motion_recoverable,
    ),
    Scenario(
        key="motion_unrecoverable",
        name="Motion artifact (unrecoverable)",
        description="Localised motion disturbance over 8-11 s, high severity.",
        expectation=(
            "Same artifact class, worse severity. Filtering raises the score "
            "but should NOT pass revalidation, so the region stays rejected "
            "while measurements are rebuilt from the rest of the record."
        ),
        build=_motion_unrecoverable,
    ),
    Scenario(
        key="severe",
        name="Severe corruption",
        description="Electrode contact lost over 3-24 s, plus a motion burst at 25-29 s.",
        expectation=(
            "Recovery should fail and stay failed. Too little trustworthy "
            "signal remains, and the system should refuse measurements rather "
            "than produce numbers from it."
        ),
        build=_severe,
    ),
]

SCENARIOS_BY_KEY: dict[str, Scenario] = {s.key: s for s in SCENARIOS}
SCENARIO_NAMES: list[str] = [s.name for s in SCENARIOS]


def scenario_by_name(name: str) -> Scenario:
    """Look a scenario up by its display name."""
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(f"no scenario named {name!r}")


def build(key: str) -> ECGRecord:
    """Build the record for a scenario key."""
    if key not in SCENARIOS_BY_KEY:
        raise KeyError(f"no scenario with key {key!r}")
    return SCENARIOS_BY_KEY[key].record()
