"""Remove complete SFT source dialogs listed in a leakage audit manifest.

The filter is intentionally lossless for retained rows: it copies original
JSONL bytes in their original order and never rewrites message content.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cleanup_set(manifest_path: Path, set_name: str) -> tuple[set[str], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set_name not in manifest:
        available = ", ".join(sorted(manifest))
        raise KeyError(f"Unknown cleanup set {set_name!r}. Available: {available}")
    selected = manifest[set_name]
    dialog_ids = selected.get("dialog_ids")
    if not isinstance(dialog_ids, list) or not dialog_ids:
        raise ValueError(f"Manifest set {set_name!r} has no non-empty dialog_ids list")
    if len(dialog_ids) != len(set(dialog_ids)):
        raise ValueError(f"Manifest set {set_name!r} contains duplicate dialog IDs")
    expected_dialogs = selected.get("source_dialogs")
    if expected_dialogs is not None and expected_dialogs != len(dialog_ids):
        raise ValueError(
            f"Manifest source_dialogs={expected_dialogs}, but dialog_ids contains {len(dialog_ids)} IDs"
        )
    return set(dialog_ids), selected


def parse_dialog_id(raw_line: bytes, line_number: int, path: Path) -> str:
    try:
        row = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON at {path}:{line_number}") from exc
    dialog_id = row.get("metadata", {}).get("source_dialog_id")
    if not isinstance(dialog_id, str) or not dialog_id:
        raise ValueError(f"Missing metadata.source_dialog_id at {path}:{line_number}")
    return dialog_id


def verify_output(path: Path, removed_ids: set[str], expected_rows: int) -> dict[str, Any]:
    rows = 0
    source_dialogs: set[str] = set()
    forbidden_ids: set[str] = set()
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            dialog_id = parse_dialog_id(raw_line, line_number, path)
            rows += 1
            source_dialogs.add(dialog_id)
            if dialog_id in removed_ids:
                forbidden_ids.add(dialog_id)
    if rows != expected_rows:
        raise ValueError(f"Verification row mismatch: expected {expected_rows}, found {rows}")
    if forbidden_ids:
        raise ValueError(f"Removed dialog IDs remain in output: {sorted(forbidden_ids)}")
    return {
        "rows": rows,
        "source_dialogs": len(source_dialogs),
        "forbidden_dialog_ids_remaining": 0,
        "valid_jsonl": True,
    }


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    manifest_path: Path,
    set_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    removed_ids, manifest_entry = load_cleanup_set(manifest_path, set_name)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    retained_rows = 0
    removed_rows = 0
    input_dialogs: set[str] = set()
    retained_dialogs: set[str] = set()
    removed_row_counts: collections.Counter[str] = collections.Counter()
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with input_path.open("rb") as source, temporary_path.open("wb") as destination:
            for line_number, raw_line in enumerate(source, 1):
                input_digest.update(raw_line)
                if not raw_line.strip():
                    # Preserve blank lines, but do not count them as SFT records.
                    destination.write(raw_line)
                    output_digest.update(raw_line)
                    continue
                dialog_id = parse_dialog_id(raw_line, line_number, input_path)
                input_rows += 1
                input_dialogs.add(dialog_id)
                if dialog_id in removed_ids:
                    removed_rows += 1
                    removed_row_counts[dialog_id] += 1
                    continue
                destination.write(raw_line)
                output_digest.update(raw_line)
                retained_rows += 1
                retained_dialogs.add(dialog_id)

        missing_manifest_ids = removed_ids - set(removed_row_counts)
        if missing_manifest_ids:
            raise ValueError(
                "Manifest dialog IDs were not found in input: "
                + ", ".join(sorted(missing_manifest_ids))
            )
        expected_removed_rows = manifest_entry.get("training_rows")
        if expected_removed_rows is not None and removed_rows != expected_removed_rows:
            raise ValueError(
                f"Manifest expects {expected_removed_rows} removed rows, but input produced {removed_rows}"
            )
        if input_rows != retained_rows + removed_rows:
            raise AssertionError("Input row accounting failed")
        if retained_dialogs & removed_ids:
            raise AssertionError("A removed source dialog remains in retained_dialogs")
        if len(input_dialogs) - len(retained_dialogs) != len(removed_ids):
            raise ValueError("Source-dialog accounting does not match the manifest")

        verification = verify_output(temporary_path, removed_ids, retained_rows)
        if output_path.exists():
            output_path.unlink()
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "drop_complete_source_dialogs",
        "cleanup_set": set_name,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_digest.hexdigest(),
            "rows": input_rows,
            "source_dialogs": len(input_dialogs),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "source_dialogs_requested": len(removed_ids),
        },
        "removed": {
            "rows": removed_rows,
            "source_dialogs": len(removed_row_counts),
            "dialog_ids": sorted(removed_ids, key=lambda value: int(value.rsplit("_", 1)[1])),
            "rows_by_dialog": dict(sorted(removed_row_counts.items())),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": output_digest.hexdigest(),
            "rows": retained_rows,
            "source_dialogs": len(retained_dialogs),
        },
        "verification": verification,
        "retained_rows_copied_without_json_rewrite": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--set-name", default="recommended_strict_primary_cleanup")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = filter_jsonl(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        manifest_path=args.manifest,
        set_name=args.set_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
