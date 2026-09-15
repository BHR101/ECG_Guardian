"""Run the five-act judge walkthrough headlessly.

    python tools/demo_flow.py              # all five acts
    python tools/demo_flow.py --act 4      # just the differential-verdict act

Each act runs the real pipeline on an existing scenario -- the same
``run_pipeline`` call the Streamlit dashboard makes.  There is no demo-only
code path, no stored result and no bypass: if the architecture stopped
producing the behaviour an act describes, this script would show it.

Use it to rehearse the demo, to check the story still holds after a change, or
to present the argument in a terminal when a browser is not available.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from src.demo import DEMO_FLOW  # noqa: E402
from src.evidence import ACCEPTED  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

WIDTH = 78

# The narration is written for the web dashboard, so it contains typographic
# punctuation.  A Windows console on a legacy code page renders those as "?",
# which looks like a bug during a demo.  Transliterate on the way out rather
# than degrading the text everything else uses.
_ASCII = {
    "—": "-", "–": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "→": "->",
    "±": "+/-", "≥": ">=", "≤": "<=",
}


def say(text: str = "") -> None:
    """Print, surviving consoles that cannot encode the source punctuation."""
    for src, dst in _ASCII.items():
        text = text.replace(src, dst)
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - last-resort fallback
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, "replace").decode(enc, "replace"))


def wrap(text: str, indent: str = "  ") -> str:
    import textwrap
    return "\n".join(
        textwrap.fill(line, WIDTH - len(indent), initial_indent=indent,
                      subsequent_indent=indent)
        for line in text.split("\n")
    )


def run_act(act, show_log: bool = False) -> dict:
    scenario = act.scenario
    result = run_pipeline(scenario.record(), scenario.name)

    say()
    say("=" * WIDTH)
    say(f"ACT {act.number} of {len(DEMO_FLOW)} — {act.title}")
    say("=" * WIDTH)
    say(f"  scenario: {scenario.name}")
    say()
    say(wrap(act.narration))
    say()
    say(wrap(f"Where to look — {act.look_at}"))
    say()
    say("-" * WIDTH)

    say(f"  signal quality      : {result.quality_raw.score:.1f} / 100 raw"
          f"  ->  {result.quality_processed.score:.1f} / 100 after recovery")
    say(f"  overall trust state : {result.overall_trust}")
    say(f"  recovery            : {result.recovery.status}")

    if result.detections:
        say("  artifacts localised :")
        for d in result.detections:
            say(f"      {d.label} — {d.start:.1f}–{d.end:.1f} s "
                  f"(confidence {d.confidence:.0%})")
    else:
        say("  artifacts localised : none above the detection threshold")

    if result.recovery.regions:
        say("  revalidation        :")
        for r in result.recovery.regions:
            mark = "VALIDATED" if r.recovered else "FAILED — segment stays rejected"
            say(f"      {r.start:.1f}–{r.end:.1f} s  "
                  f"{r.score_before:.0f} -> {r.score_after:.0f}  {mark}")

    say()
    say("  MEASUREMENT VERDICTS")
    for m in result.measurements:
        if m.status == ACCEPTED:
            say(f"      {m.display_name:<16} ACCEPTED      {m.display_value:>12}"
                  f"   ({m.confidence:.0%} confidence)")
        else:
            say(f"      {m.display_name:<16} NOT REPORTED  {'—':>12}"
                  "   evidence insufficient")
            for c in m.failed_criteria[:2]:
                say(f"          failed: {c.name} was {c.value} "
                      f"(needed {c.required})")
            if m.withheld_value is not None:
                say(f"          withheld: the arithmetic produced "
                      f"{m.withheld_value:g} {m.unit} — not reported, because "
                      "the evidence behind it did not survive")

    if show_log:
        say()
        say("  PIPELINE LOG")
        for e in result.log:
            say(f"      [{e.stage:<12}] {e.message}")

    if result.errors:
        say(f"\n  stage errors: {'; '.join(result.errors)}")

    return {
        "act": act.number,
        "accepted": [m.measurement for m in result.measurements
                     if m.status == ACCEPTED],
        "not_reported": [m.measurement for m in result.measurements
                         if m.status != ACCEPTED],
        "errors": list(result.errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", type=int, default=None,
                    help="run a single act (1-5) instead of all five")
    ap.add_argument("--log", action="store_true",
                    help="also print the pipeline event log for each act")
    args = ap.parse_args()

    acts = DEMO_FLOW
    if args.act is not None:
        match = [a for a in DEMO_FLOW if a.number == args.act]
        if not match:
            say(f"No act {args.act}; acts are 1-{len(DEMO_FLOW)}.")
            return 1
        acts = match

    say()
    say("ECG GUARDIAN — five-act walkthrough")
    say("Every number below comes from running the real pipeline now.")
    outcomes = [run_act(a, show_log=args.log) for a in acts]

    say()
    say("=" * WIDTH)
    say("THE STORY IN ONE TABLE")
    say("=" * WIDTH)
    say(f"  {'act':<5}{'heart rate':<16}{'QRS duration':<16}{'title'}")
    for o, a in zip(outcomes, acts):
        hr = "ACCEPTED" if "heart_rate" in o["accepted"] else "NOT REPORTED"
        qrs = "ACCEPTED" if "qrs_duration" in o["accepted"] else "NOT REPORTED"
        say(f"  {a.number:<5}{hr:<16}{qrs:<16}{a.title[:34]}")
    say()
    say(wrap(
        "The row that matters is the act where the two columns disagree. A "
        "single global quality verdict cannot produce that row: it is what "
        "measurement-specific evidence gating is for.", indent="  "))

    errored = [o for o in outcomes if o["errors"]]
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
