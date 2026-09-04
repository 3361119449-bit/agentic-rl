"""Stage -1 convenience wrapper: eight rollouts per official train task."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("evaluate_airline.py")
    project_root = Path(__file__).resolve().parents[1]
    tag = "sft_baseline_profile"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--split",
            "official_train",
            "--samples",
            "8",
            "--tag",
            tag,
            *sys.argv[1:],
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("profile_rollouts.py")),
            str(project_root / "outputs" / "evaluations" / tag / "trajectories"),
            "--group-size",
            "8",
            "--output",
            str(project_root / "outputs" / "reports" / f"{tag}.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
