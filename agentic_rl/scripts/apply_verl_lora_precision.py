"""Apply or verify the FP32 LoRA and micro-batch patches for pinned RL veRL."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

PATCH_DIR = Path(__file__).resolve().parents[1] / "patches"
PATCHES = ("verl-fp32-lora-master.patch", "verl-strict-microbatch-budget.patch")
MUON = "verl-fp32-lora-muon.patch"
ENGINE = Path("verl/workers/engine/fsdp/transformer_impl.py")
PIN = "483b8a009ba3a97563edee3a19887e4862b8094a"


def patch_command(root: Path, filename: str, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), "apply", *flags, str(PATCH_DIR / filename)],
        capture_output=True,
        text=True,
    )


def installed(root: Path, filename: str) -> bool:
    if not patch_command(root, filename, "--reverse", "--check").returncode:
        return True
    if filename != PATCHES[0] or patch_command(root, MUON, "--reverse", "--check").returncode:
        return False
    # The Muon add-on extends two original AdamW hunks. Verify the underlying
    # master-weight patch on a private copy without touching the live engine.
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp)
        (copy / ENGINE).parent.mkdir(parents=True)
        shutil.copyfile(root / ENGINE, copy / ENGINE)
        return (not patch_command(copy, MUON, "--reverse").returncode
                and not patch_command(copy, filename, "--reverse", "--check").returncode)


def require_precision_patch(root: Path, *, muon: bool = False) -> None:
    expected_patches = (*PATCHES, MUON) if muon else PATCHES
    for filename in expected_patches:
        if not installed(root, filename):
            raise RuntimeError(
                f"Missing pinned veRL patch: {filename}. Run python "
                f"scripts/apply_verl_lora_precision.py {'--muon ' if muon else ''}<verl_root>."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_root", type=Path)
    parser.add_argument("--muon", action="store_true", help="Also enable native Muon for single-GPU FP32 LoRA.")
    args = parser.parse_args()
    root = args.verl_root.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise ValueError(f"Expected veRL {PIN}, got {head}")
    missing = [
        filename for filename in PATCHES
        if not installed(root, filename)
    ]
    for filename in missing:
        result = patch_command(root, filename, "--check")
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    for filename in missing:
        result = patch_command(root, filename)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    if args.muon and not installed(root, MUON):
        result = patch_command(root, MUON)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
    require_precision_patch(root, muon=args.muon)
    print(f"FP32 LoRA and micro-batch patches verified in {root}")


if __name__ == "__main__":
    main()
