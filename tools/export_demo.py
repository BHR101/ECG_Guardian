"""Write each demo scenario's waveform and verdicts into data/generated/.

Useful for sharing a fixed record, or for inspecting a
run outside the dashboard.  The dashboard itself needs none of this -- the
pipeline runs entirely in memory.

    python tools/export_demo.py
"""

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from src.demo import SCENARIOS  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "generated"
)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        record = scenario.record()
        result = run_pipeline(record, scenario.name)

        frame = pd.DataFrame(
            {
                "time_s": record.time,
                "raw_mv": record.signal,
                "processed_mv": result.processed_signal,
            }
        )
        csv_path = os.path.join(OUT_DIR, f"{scenario.key}.csv")
        frame.to_csv(csv_path, index=False, float_format="%.6f")

        summary = result.summary()
        summary["scenario_key"] = scenario.key
        summary["description"] = scenario.description
        json_path = os.path.join(OUT_DIR, f"{scenario.key}.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        accepted = ", ".join(m.measurement for m in result.accepted) or "none"
        print(
            f"{scenario.key:22s} -> {os.path.basename(csv_path):28s} "
            f"trust={result.overall_trust:9s} accepted: {accepted}"
        )

    print(f"\nWritten to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
