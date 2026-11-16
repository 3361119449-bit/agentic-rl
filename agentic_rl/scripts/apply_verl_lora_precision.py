"""Apply or verify the FP32 LoRA and micro-batch patches for pinned RL veRL."""

import argparse
import subprocess
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"
PATCHES = ("verl-fp32-lora-master.patch", "verl-strict-microbatch-budget.patch")
PIN = "483b8a009ba3a97563edee3a19887e4862b8094a"


def patch_command(root: Path, filename: str, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "apply", *flags, str(PATCH_DIR / filename)],
        capture_output=True,
        text=True,
    )


def require_precision_patch(root: Path) -> None:
    for filename in PATCHES:
        if patch_command(root, filename, "--reverse", "--check").returncode:
            raise RuntimeError(
                f"Missing pinned veRL patch: {filename}. Run python "
                "scripts/apply_verl_lora_precision.py with the veRL checkout path."
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
    missing = [
        filename for filename in PATCHES
        if patch_command(root, filename, "--reverse", "--check").returncode
    ]
    for filename in missing:
        result = patch_command(root, filename, "--check")
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    for filename in missing:
        result = patch_command(root, filename)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    require_precision_patch(root)
    print(f"FP32 LoRA and micro-batch patches verified in {root}")


if __name__ == "__main__":
    main()
