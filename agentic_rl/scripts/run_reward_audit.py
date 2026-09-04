"""Compare hand-calculated audit expectations with saved program scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_file", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    rows = json.loads(args.audit_file.read_text(encoding="utf-8"))
    failures = []
    for row in rows:
        trajectory_path = Path(row["trajectory"])
        record = json.loads(trajectory_path.read_text(encoding="utf-8"))
        actual = float(record["custom_reward"]["train_reward"])
        expected = float(row["expected_train_reward"])
        if abs(actual - expected) > args.tolerance:
            failures.append(
                {
                    "trajectory": str(trajectory_path),
                    "expected": expected,
                    "actual": actual,
                }
            )
    print(json.dumps({"checked": len(rows), "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
