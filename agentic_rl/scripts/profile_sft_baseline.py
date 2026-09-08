"""Stage -1 convenience wrapper: eight rollouts per official train task."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    from scripts.evaluate_airline import parse_args
except ModuleNotFoundError:  # Direct python scripts/profile_sft_baseline.py.
    from evaluate_airline import parse_args


def main() -> None:
    script = Path(__file__).with_name("evaluate_airline.py")
    project_root = Path(__file__).resolve().parents[1]
    arguments = [
        "--split",
        "official_train",
        "--samples",
        "8",
        "--tag",
        "sft_baseline_profile",
        *sys.argv[1:],
    ]
    args = parse_args(arguments)
    subprocess.run(
        [
            sys.executable,
            str(script),
            *arguments,
        ],
        check=True,
    )
    if args.dry_run:
        return  # No new trajectories exist, and old reports must not be touched.
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("profile_rollouts.py")),
            str(project_root / "outputs" / "evaluations" / args.tag / "trajectories"),
            "--group-size",
            str(args.samples),
            "--output",
            str(project_root / "outputs" / "reports" / f"{args.tag}.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
