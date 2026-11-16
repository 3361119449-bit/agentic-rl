"""Apply or verify the FP32 LoRA master patch for the pinned RL veRL checkout."""

import argparse
import subprocess
from pathlib import Path

PATCH = Path(__file__).resolve().parents[1] / "patches" / "verl-fp32-lora-master.patch"
PIN = "483b8a009ba3a97563edee3a19887e4862b8094a"


def patch_command(root: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "apply", *flags, str(PATCH)],
        capture_output=True,
        text=True,
    )


def require_precision_patch(root: Path) -> None:
    if patch_command(root, "--reverse", "--check").returncode:
        raise RuntimeError(
            "FP32 LoRA requires the pinned veRL precision patch. "
            "Run python scripts/apply_verl_lora_precision.py with the veRL checkout path."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_root", type=Path)
    args = parser.parse_args()
    root = args.verl_root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise ValueError(f"Expected veRL {PIN}, got {head}")
    if patch_command(root, "--reverse", "--check").returncode:
        result = patch_command(root, "--check")
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        result = patch_command(root)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    require_precision_patch(root)
    print(f"FP32 LoRA patch verified in {root}")


if __name__ == "__main__":
    main()
