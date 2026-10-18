"""Compare optional local credit on original audited groups, with no model calls.

The dependency-light fixed-tool harness executes the pinned, unmodified airline
tool method bodies and Pydantic domain models. It substitutes only transport,
logging and toolkit initialization; it is not a full orchestrator/GPU smoke.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

from pydantic import BaseModel, ConfigDict

from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.procredit_runtime import credit_from_record, queue_group_reports
from tau2_agentic_rl.reward.db_subgoals import (
    DB_SUBGOAL_VERSION,
    select_db_subgoals,
    subgoal_trace_from_scoring,
)
from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS, _event_satisfies
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.versions import sha256_json

PINNED_TAU2 = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"


def load_fixed_tools(source: Path):
    """Load exact pinned source bodies without importing Tau2's LLM runner."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args])

    if git("rev-parse", "HEAD").decode().strip() != PINNED_TAU2:
        raise ValueError("offline replay requires the pinned Tau2 commit")
    domain = "src/tau2/domains/airline/"
    data_path = source / "data/tau2/domains/airline/db.json"
    if (
        git("show", f"{PINNED_TAU2}:data/tau2/domains/airline/db.json")
        != data_path.read_bytes()
    ):
        raise ValueError("pinned initial airline database changed")
    module = ModuleType("_db_subgoal_pinned_airline")
    sys.modules[module.__name__] = module

    class DB(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Toolkit:
        def __init__(self, db):
            self.db = db

    module.__dict__.update(
        DB=DB,
        ToolKitBase=Toolkit,
        AIRLINE_DB_PATH=data_path,
        logger=SimpleNamespace(
            **{name: lambda *a, **kw: None for name in ("info", "debug", "warning")}
        ),
    )
    for filename in ("data_model.py", "tools.py"):
        path = domain + filename
        raw = git("show", f"{PINNED_TAU2}:{path}")
        tree = ast.parse(raw.decode(), filename=path)
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and (node.module.startswith("tau2.") or node.module == "loguru")
            )
            and not isinstance(node, ast.If)
        ]
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
        exec(compile(tree, path, "exec"), module.__dict__)
    return module.AirlineTools, module.FlightDB.model_validate_json(
        data_path.read_text(encoding="utf-8")
    )


def replay_initial(record, tools_type, base_db):
    """Verify every returned write snapshot against the actual pinned DB object."""
    frozen = record.scoring_inputs
    task = frozen["judge"]["task"]
    goals = select_db_subgoals(
        task,
        frozen["required_actions"],
        frozen["action_dependencies"],
        frozen["judge"]["transfer_rule"],
    )
    if not goals:
        return {}, 0, None
    if task.get("initial_state") is not None or task.get("user_tools"):
        return {}, 0, "unsupported initial/user state in offline harness"
    tools = tools_type(base_db.model_copy(deep=True))
    ids = {a["arguments"]["reservation_id"] for a in goals}
    initial = {
        rid: tools.db.reservations[rid].model_dump(mode="json")
        for rid in ids
        if rid in tools.db.reservations
    }
    checked = 0
    for event in record.tool_events:
        if event.name not in MUTATING_TOOLS or event.execution_status is not None:
            continue
        if event.error_kind in {
            "parse_error",
            "unknown_tool",
            "schema_invalid",
            "mixed_content_and_tool_call",
            "multiple_tool_calls",
        }:
            continue
        before = tools.db.model_dump(mode="json")
        if (
            event.name == "update_reservation_baggages"
            and event.state_before is not None
        ):
            reservation = tools.db.reservations.get(
                event.arguments.get("reservation_id")
            )
            expected_before = (
                {
                    "reservation_id": reservation.reservation_id,
                    "nonfree_baggages": reservation.nonfree_baggages,
                }
                if reservation is not None
                else None
            )
            if event.state_before != expected_before:
                return (
                    {},
                    checked,
                    f"{event.event_id}: baggage state_before differs from pinned DB",
                )
        try:
            result = getattr(tools, event.name)(**event.arguments)
        except (ValueError, TypeError) as exc:
            if event.success:
                return {}, checked, f"{event.event_id}: {type(exc).__name__}: {exc}"
        else:
            if not event.success:
                return {}, checked, f"{event.event_id}: replay unexpectedly succeeded"
            actual = (
                result.model_dump(mode="json")
                if isinstance(result, BaseModel)
                else result
            )
            try:
                recorded = json.loads(event.result)
            except (ValueError, TypeError):
                recorded = event.result
            if actual != recorded:
                return (
                    {},
                    checked,
                    f"{event.event_id}: returned snapshot differs from pinned replay",
                )
            rid = event.arguments.get("reservation_id")
            if rid in ids:
                if actual != tools.db.reservations[rid].model_dump(mode="json"):
                    return (
                        {},
                        checked,
                        f"{event.event_id}: returned snapshot is not current DB state",
                    )
                checked += 1
        changed = before != tools.db.model_dump(mode="json")
        if event.db_effect is not changed:
            return (
                {},
                checked,
                f"{event.event_id}: DB effect differs from pinned replay",
            )
    return initial, checked, None


def compare_group(report, records, initial_for_record):
    """Keep every original member and compare token values before/after opt-in."""
    terminal = (
        {report["uid"]: report["terminal_group"]}
        if report["version"] == "procredit-group-v2"
        else None
    )
    try:
        original = queue_group_reports(
            report["members"],
            [{"procredit": r} for r in report["inputs"]],
            CreditConfig(**report["credit"]["config"]),
            terminal_groups=terminal,
        )
    except ValueError as exc:
        raise ValueError("original group membership changed") from exc
    if original != {report["uid"]: report}:
        raise ValueError("original group membership, fingerprints or credit changed")
    old = compute_group_credit(
        report["inputs"], CreditConfig(**report["credit"]["config"])
    )
    if old != report["credit"]:
        raise ValueError("disabled credit does not reproduce the original Group Audit")
    rows, traces, source_rows = [], [], []
    for original, record in zip(report["inputs"], records, strict=True):
        if credit_from_record(record) != original:
            raise ValueError("saved record differs from original audited credit row")
        initial, checked, error = initial_for_record(record)
        frozen = deepcopy(record.scoring_inputs)
        frozen["reward_project_config"]["credit"]["db_subgoal_credit"] = (
            DB_SUBGOAL_VERSION
        )
        frozen["db_subgoal_inputs"] = {
            "initial_reservations": initial,
            "initial_state_fingerprint": record.progress_trace[
                "initial_state_fingerprint"
            ],
        }
        trace = subgoal_trace_from_scoring(
            frozen, record.tool_events, record.assistant_turns, record.custom_reward
        )
        updated = record.model_copy(
            update={
                "scoring_inputs": frozen,
                "subgoal_trace": trace,
                "metadata": record.metadata
                | {"scoring_inputs_sha256": sha256_json(frozen)},
            }
        )
        row = credit_from_record(updated)
        assert row["score"] == original["score"] and row["phi"] == original["phi"]
        rows.append(row)
        traces.append(trace)
        source_rows.append({"verified_snapshots": checked, "replay_exclusion": error})
    cfg = CreditConfig(
        **(report["credit"]["config"] | {"db_subgoal_credit": DB_SUBGOAL_VERSION})
    )
    new = compute_group_credit(rows, cfg)
    assert all(
        new[k] == old[k]
        for k in ("score_mean", "score_std", "turn_mean", "valid_turns")
    )
    details, mistakes = [], []
    for original, record, trace, prior, after, verification in zip(
        report["inputs"],
        records,
        traces,
        old["trajectories"],
        new["trajectories"],
        source_rows,
        strict=True,
    ):
        by_turn = {}
        for t, before, now in zip(
            original["response_turn_ids"],
            prior["token_advantages"],
            after["token_advantages"],
            strict=True,
        ):
            if t < 0:
                assert before == now == 0
            else:
                by_turn[t] = (before, now)
        wrong = {
            e.turn_id - 1
            for e in record.tool_events
            if e.name in MUTATING_TOOLS
            and not any(
                _event_satisfies(a, e)
                for a in record.scoring_inputs["required_actions"]
            )
        }
        policy = set(original["policy_credit"]["violating_turns"])
        for t in policy:
            assert by_turn[t][1] <= -1 - original["process_credit"]["turn_costs"][t]
        for t in wrong:
            before, now = by_turn[t]
            if before <= 1e-8 and now > 1e-8:
                mistakes.append(
                    dict(
                        trajectory_id=record.trajectory_id,
                        task_id=record.task_id,
                        turn_id=t + 1,
                        before=before,
                        after=now,
                    )
                )
        details.append(
            {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "terminal_success": original["terminal_success"],
                "train_reward": original["score"],
                "subgoal_trace": trace,
                **verification,
                "turns": [
                    dict(
                        turn_id=t + 1,
                        before=b,
                        after=n,
                        delta=n - b,
                        wrong_write=t in wrong,
                        policy_violation=t in policy,
                    )
                    for t, (b, n) in sorted(by_turn.items())
                ],
                "token_advantages_before_sha256": sha256_json(
                    prior["token_advantages"]
                ),
                "token_advantages_after_sha256": sha256_json(after["token_advantages"]),
            }
        )
    return {
        "uid": report["uid"],
        "members": report["members"],
        "source_audit_fingerprint": report["audit_fingerprint"],
        "credit_config_before": old["config"],
        "credit_config_after": new["config"],
        "before_has_signal": old["has_signal"],
        "after_has_signal": new["has_signal"],
        "trajectories": details,
        "misrewarded_turns": mistakes,
    }


def audit(source: Path, output: Path, tau2_source: Path):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("audit output must be a fresh directory")
    if source == output or output in source.parents or source in output.parents:
        raise ValueError("audit output must be separate from source")
    tools_type, db = load_fixed_tools(tau2_source)
    paths = sorted((source / "group_audits").glob("*.json"))
    if not paths:
        raise ValueError("no original group audits")
    output.mkdir(parents=True)
    (output / "groups").mkdir()
    counts, tasks, errors, mistakes, ids = Counter(), Counter(), [], [], set()
    for index, path in enumerate(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        payload = {k: v for k, v in report.items() if k != "audit_fingerprint"}
        if report.get("audit_fingerprint") != sha256_json(payload) or report.get(
            "inputs_fingerprint"
        ) != sha256_json(report["inputs"]):
            raise ValueError(f"invalid group fingerprint: {path}")
        records = []
        for row in report["inputs"]:
            trajectory_id = row["trajectory_id"]
            if (
                len(trajectory_id) != 32
                or any(c not in "0123456789abcdef" for c in trajectory_id)
                or trajectory_id in ids
            ):
                raise ValueError("invalid or reused trajectory identity")
            ids.add(trajectory_id)
            records.append(
                TrajectoryRecord.model_validate_json(
                    (source / "trajectories" / f"{trajectory_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        result = compare_group(
            report, records, lambda r: replay_initial(r, tools_type, db)
        )
        counts["groups"] += 1
        counts["zero_groups_rescued"] += (
            not result["before_has_signal"] and result["after_has_signal"]
        )
        counts["groups_without_positive_rescued"] += not any(
            t["before"] > 1e-8 for r in result["trajectories"] for t in r["turns"]
        ) and any(t["after"] > 1e-8 for r in result["trajectories"] for t in r["turns"])
        mistakes.extend(result["misrewarded_turns"])
        for r in result["trajectories"]:
            trace = r["subgoal_trace"]
            counts["trajectories"] += 1
            counts["eligible_trajectories"] += trace["applicable"]
            if trace["applicable"]:
                tasks[r["task_id"]] += 1
            counts["failed_trajectories_with_partial_progress"] += not r[
                "terminal_success"
            ] and any(trace["phi"])
            counts["turns_with_positive_local_credit"] += sum(
                v > 0 for v in trace["turn_credits"]
            )
            counts["correct_turns_newly_positive"] += sum(
                t["before"] <= 1e-8 and t["after"] > 1e-8 for t in r["turns"]
            )
            counts["deferred_mixed_or_policy_turns"] += len(trace["deferred_turns"])
            counts["regression_turns"] += len(trace["regression_turns"])
            counts["verified_snapshots"] += r["verified_snapshots"]
            counts["policy_violation_turns_checked"] += sum(
                t["policy_violation"] for t in r["turns"]
            )
            if r["replay_exclusion"]:
                errors.append(
                    {
                        "trajectory_id": r["trajectory_id"],
                        "reason": r["replay_exclusion"],
                    }
                )
        (output / "groups" / path.name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if (index + 1) % 12 == 0:
            print(f"audited {index + 1}/{len(paths)} groups", flush=True)
    summary = dict(counts) | {
        "eligible_tasks": dict(tasks),
        "baseline_token_max_abs_error": 0.0,
        "misrewarded_wrong_write_turns": mistakes,
        "replay_exclusions": errors,
        "fixed_tau2_commit": PINNED_TAU2,
        "replay_harness": "pinned method bodies and Pydantic models; no orchestrator",
        "acceptance_passed": not mistakes and not errors,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tau2-source", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.output, args.tau2_source)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["acceptance_passed"]:
        raise SystemExit("DB subgoal offline acceptance failed; inspect summary.json")


if __name__ == "__main__":
    main()
