"""Recompute original audited membership into a fresh directory, with no model calls."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from tau2_agentic_rl.advantages import CreditConfig
from tau2_agentic_rl.config import load_yaml, validate_procredit_config
from tau2_agentic_rl.procredit_runtime import credit_from_record, queue_group_reports
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.versions import sha256_json


def rescore_groups(
    source: Path,
    output: Path,
    project: dict,
    *,
    records_dir: Path | None = None,
) -> int:
    """Keep membership fixed, optionally use corresponding newly rescored records."""
    validate_procredit_config(project)
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("group rescoring requires a separate new output directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("group output directory must be empty")
    prepared, uids = [], set()
    for path in sorted(source.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = report.pop("audit_fingerprint", None)
        if (
            report.get("version") not in {"procredit-group-v1", "procredit-group-v2"}
            or fingerprint != sha256_json(report)
            or report.get("inputs_fingerprint") != sha256_json(report.get("inputs"))
        ):
            raise ValueError(f"invalid original group audit: {path}")
        uid = report["uid"]
        if uid in uids:
            raise ValueError("duplicate original sampling uid")
        uids.add(uid)
        terminal_groups = (
            {uid: report.get("terminal_group")}
            if report["version"] == "procredit-group-v2" else None
        )
        original = queue_group_reports(
            report["members"],
            [{"procredit": row} for row in report["inputs"]],
            CreditConfig(**report["credit"]["config"]),
            terminal_groups=terminal_groups,
        )
        if set(original) != {uid} or original[uid] != {**report, "audit_fingerprint": fingerprint}:
            raise ValueError("original group membership or credit changed")
        rows = deepcopy(report["inputs"])
        if records_dir is not None:
            root = records_dir.resolve()
            for index, old in enumerate(rows):
                record_path = (root / f"{old['trajectory_id']}.json").resolve()
                if record_path.parent != root:
                    raise ValueError("unsafe trajectory path in group audit")
                record = TrajectoryRecord.model_validate_json(
                    record_path.read_text(encoding="utf-8")
                )
                new = credit_from_record(record)
                for field in (
                    "trajectory_id",
                    "task_id",
                    "policy_version",
                    "initial_state_fingerprint",
                    "response_turn_ids",
                ):
                    if new[field] != old[field]:
                        raise ValueError(
                            "rescored record differs from original interaction"
                        )
                rows[index] = new
        # Validate every input before creating any output.
        queue_group_reports(
            report["members"],
            [{"procredit": row} for row in rows],
            CreditConfig.from_project(project),
            terminal_groups=terminal_groups,
        )
        prepared.append((report["members"], rows, terminal_groups))
    if not prepared:
        raise ValueError("no ProCredit group audits found")
    for keys, rows, terminal_groups in prepared:
        queue_group_reports(
            keys,
            [{"procredit": row} for row in rows],
            CreditConfig.from_project(project),
            audit_dir=output,
            terminal_groups=terminal_groups,
        )
    return len(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path)
    args = parser.parse_args()
    count = rescore_groups(
        args.source, args.output, load_yaml(args.config), records_dir=args.records_dir
    )
    print(f"rescored {count} original groups into {args.output}")


if __name__ == "__main__":
    main()
