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

# ---------------------------------------------------------------------------
# The guided walkthrough
# ---------------------------------------------------------------------------
# Five acts that tell one story, in order.  Every act names an existing
# scenario -- there is no separate demo path, no stored result and no bypass:
# running an act runs the same ``run_pipeline`` call the dashboard always runs.
# The narration describes what to look at; the numbers come from the pipeline.
@dataclass
class DemoAct:
    """One step of the guided walkthrough."""

    number: int
    title: str
    scenario_key: str
    narration: str
    look_at: str          # where on the dashboard the point is visible

    @property
    def scenario(self) -> Scenario:
        return SCENARIOS_BY_KEY[self.scenario_key]


DEMO_FLOW: list[DemoAct] = [
    DemoAct(
        number=1,
        title="Clean ECG — evidence is sufficient",
        scenario_key="clean",
        narration=(
            "This is a clean recording. The evidence supports both timing and "
            "morphology measurements, so every measurement is reported."
        ),
        look_at="Measurements: all four accepted, with the criteria each one met.",
    ),
    DemoAct(
        number=2,
        title="Corrupted ECG — locating the unreliable evidence",
        scenario_key="motion_unrecoverable",
        narration=(
            "Now the signal is corrupted. Instead of blindly filtering and "
            "trusting the result, we identify where the evidence has become "
            "unreliable. The disturbance is localised to its own span, and that "
            "span is labelled — not silently smoothed into the rest."
        ),
        look_at=(
            "Artifact intelligence: the localised region. Trust map: that span "
            "is rejected while the rest of the record stays usable."
        ),
    ),
    DemoAct(
        number=3,
        title="Recovery and mandatory revalidation",
        scenario_key="motion_recoverable",
        narration=(
            "Recovery is not considered successful just because filtering was "
            "applied. The artifact-defining property is checked again. This is "
            "the same artifact class as Act 2 at a lower severity: here "
            "revalidation passes and the region is allowed forward, whereas in "
            "Act 2 the score rose but revalidation still refused it. That "
            "contrast is the point — revalidation is not a rubber stamp."
        ),
        look_at=(
            "Recovery and revalidation: the two independent tests, and the "
            "region's before/after verdict."
        ),
    ),
    DemoAct(
        number=4,
        title="One ECG, two different verdicts",
        scenario_key="muscle_noise",
        narration=(
            "This is the key idea. We do not ask whether the ECG is simply "
            "'good' or 'bad'. We ask whether there is enough evidence for each "
            "measurement. Heart rate only needs reliable timing evidence, so it "
            "can be accepted. QRS duration depends on morphology that is not "
            "sufficiently reliable here, so it is deliberately not reported — "
            "from the very same beats."
        ),
        look_at=(
            "Measurements: heart rate ACCEPTED, QRS duration NOT REPORTED, on "
            "one record. A single global quality gate cannot produce that."
        ),
    ),
    DemoAct(
        number=5,
        title="Severe corruption — the correct output is no report",
        scenario_key="severe",
        narration=(
            "When the evidence cannot be recovered, the correct output is not a "
            "confident-looking number. It is no report. The arithmetic still "
            "produces a value here; the system withholds it because the "
            "evidence behind it did not survive."
        ),
        look_at=(
            "Measurements: every measurement NOT REPORTED, each naming the "
            "criterion that failed, with the withheld value shown as withheld."
        ),
    ),
]

DEMO_FLOW_BY_NUMBER: dict[int, DemoAct] = {a.number: a for a in DEMO_FLOW}


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
