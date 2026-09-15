"""Run the adversarial corpus and report where the system can be fooled.

    python tools/adversarial_run.py             # full corpus
    python tools/adversarial_run.py --random 0  # hand-designed cases only
    python tools/adversarial_run.py --family contact
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from src.adversarial import (  # noqa: E402
    FALSE_ACCEPT,
    FALSE_REJECT,
    TRUE_ACCEPT,
    TRUE_REJECT,
    all_cases,
    judge_case,
)
from src.pipeline import run_pipeline  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "generated", "adversarial_results.csv",
)


def run(n_random: int, family: str | None) -> pd.DataFrame:
    rows = []
    for case in all_cases(include_random=n_random):
        if family and case.family != family:
            continue
        try:
            result = run_pipeline(case.record, case.key)
        except Exception as exc:  # a crash is itself a finding
            rows.append({
                "case": case.key, "family": case.family,
                "crashed": True, "error": str(exc),
                "hr_outcome": "CRASH", "qrs_outcome": "CRASH",
            })
            continue
        row = judge_case(case, result)
        row["crashed"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--family", type=str, default=None)
    args = ap.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)

    df = run(args.random, args.family)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    print("=" * 96)
    print("ECG GUARDIAN -- ADVERSARIAL RUN")
    print("=" * 96)
    print(f"cases: {len(df)}   crashes: {int(df['crashed'].sum())}")
    if "pipeline_errors" in df:
        print(f"uncaught stage errors: {int(df['pipeline_errors'].fillna(0).sum())}")

    for key, label in (("hr", "HEART RATE"), ("qrs", "QRS DURATION")):
        col = f"{key}_outcome"
        if col not in df:
            continue
        counts = df[col].value_counts().to_dict()
        n_rep = int(df[f"{key}_reported"].fillna(False).sum())
        fa = counts.get(FALSE_ACCEPT, 0)
        print(f"\n{label}")
        print(f"    reported in            {n_rep}/{len(df)} cases")
        print(f"    TRUE_ACCEPT            {counts.get(TRUE_ACCEPT, 0)}")
        print(f"    FALSE_ACCEPT           {fa}   <-- the metric that matters")
        print(f"    TRUE_REJECT            {counts.get(TRUE_REJECT, 0)}")
        print(f"    FALSE_REJECT           {counts.get(FALSE_REJECT, 0)}")
        if n_rep:
            print(f"    false acceptance rate  {fa / n_rep:.1%} of reported values")

        bad = df[df[col] == FALSE_ACCEPT]
        if len(bad):
            print(f"\n    FALSE ACCEPTANCES ({label}):")
            cols = ["case", "family", f"{key}_value", f"{key}_true",
                    f"{key}_error", f"{key}_confidence", "recovery_status"]
            cols = [c for c in cols if c in bad.columns]
            print(bad[cols].to_string(index=False))

    # Evidence-based view, for comparison with the outcome-based one above.
    print("\nEVIDENCE-BASED VIEW (was it supportable at all?)")
    for key, label in (("hr", "heart rate"), ("qrs", "QRS duration")):
        col = f"{key}_evidence_verdict"
        if col not in df:
            continue
        c = df[col].value_counts().to_dict()
        print(f"    {label:14s} accept ok {c.get(TRUE_ACCEPT,0):4d} | "
              f"FALSE ACCEPT {c.get(FALSE_ACCEPT,0):4d} | "
              f"reject ok {c.get(TRUE_REJECT,0):4d} | "
              f"over-rejected {c.get(FALSE_REJECT,0):4d}")

    # Confidence attack: high confidence on a wrong number is the worst case.
    print("\nCONFIDENCE ON FALSE ACCEPTANCES (high = P0, low = P1)")
    for key, label in (("hr", "heart rate"), ("qrs", "QRS duration")):
        col = f"{key}_outcome"
        if col not in df:
            continue
        bad = df[df[col] == FALSE_ACCEPT]
        if not len(bad):
            print(f"    {label:14s} none")
            continue
        conf = pd.to_numeric(bad[f"{key}_confidence"], errors="coerce")
        p0 = int((conf >= 0.80).sum())
        print(f"    {label:14s} n={len(bad)}  max conf {conf.max():.2f}  "
              f"P0 (conf>=0.80): {p0}")

    print(f"\nwritten to {OUT}")
    total_fa = sum(
        int((df[f"{k}_outcome"] == FALSE_ACCEPT).sum())
        for k in ("hr", "qrs") if f"{k}_outcome" in df
    )
    return 1 if total_fa or df["crashed"].any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
