"""Standalone actor update from an already generated ProCredit v4 smoke bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The CLI runs directly from scripts/; this adds only this repository's scripts
# package, leaving the installed normal training package and entrypoints intact.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.saved_rollout_support.batch import load_saved_batch, verify_base_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results .tar.gz or extracted run directory",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Original merged SFT base; required for actual update",
    )
    parser.add_argument(
        "--verl-root",
        type=Path,
        help="Existing checkout of the project's pinned veRL commit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New, empty directory for this update and checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate saved data without loading actor or contacting APIs",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Optional JSON report destination for dry-run"
    )
    args = parser.parse_args()
    if not args.dry_run and (
        args.model_path is None or args.verl_root is None or args.output_dir is None
    ):
        parser.error(
            "actual update requires --model-path, --verl-root and --output-dir"
        )
    batch = load_saved_batch(args.results)
    if args.model_path is not None:
        verify_base_model(batch, args.model_path)
    if args.dry_run:
        report = json.dumps(batch.manifest, indent=2, ensure_ascii=False)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(report + "\n", encoding="utf-8")
        print(report)
        return
    # GPU/Ray imports happen only after the full saved-data and model checks.
    from scripts.saved_rollout_support.runtime import run_update

    run_update(
        batch,
        model_path=args.model_path,
        verl_root=args.verl_root,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
