"""Audit original PPO groups under baseline, DB-only, completion-only and both."""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from scripts.audit_db_subgoals import (
    PINNED_TAU2,
    compare_group,
    load_fixed_tools,
    replay_initial,
)
from tau2_agentic_rl.advantages import CreditConfig, compute_group_credit
from tau2_agentic_rl.procredit_runtime import credit_from_record
from tau2_agentic_rl.reward.post_completion import (
    POST_COMPLETION_VERSION,
    completion_trace_from_scoring,
)
from tau2_agentic_rl.reward.required_actions import MUTATING_TOOLS
from tau2_agentic_rl.schemas import TrajectoryRecord
from tau2_agentic_rl.versions import sha256_json


def verify_db_prefixes(record, tools_type, base_db):
    """Independently replay native gold/predicted DBs for every claimed turn."""
    task = record.scoring_inputs["judge"]["task"]
    if task.get("initial_state") is not None or task.get("user_tools"):
        raise ValueError("offline completion replay cannot verify initial/user state")
    gold = tools_type(base_db.model_copy(deep=True))
    for action in task["evaluation_criteria"].get("actions") or []:
        if action.get("requestor", "assistant") != "assistant":
            raise ValueError("unsupported golden user action")
        try:
            getattr(gold, action["name"])(**action["arguments"])
        except (ValueError, TypeError):
            pass  # Same pinned EnvironmentEvaluator behavior for golden errors.
    target = gold.db.model_dump(mode="json")
    live = tools_type(base_db.model_copy(deep=True))
    trace = record.progress_trace
    names = [c["check_id"] for c in trace["checks"]]
    expected = [s[names.index("db")] for s in trace["check_values"]]
    actual = [live.db.model_dump(mode="json") == target]
    for t in range(1, record.assistant_turns + 1):
        for event in record.tool_events:
            if (
                event.turn_id != t
                or event.name not in MUTATING_TOOLS
                or event.execution_status is not None
            ):
                continue
            if event.error_kind in {
                "parse_error",
                "unknown_tool",
                "schema_invalid",
                "mixed_content_and_tool_call",
                "multiple_tool_calls",
            }:
                continue
            before = live.db.model_dump(mode="json")
            try:
                returned = getattr(live, event.name)(**event.arguments)
            except (ValueError, TypeError):
                if event.success:
                    raise ValueError(
                        f"{event.event_id}: fixed replay unexpectedly failed"
                    )
            else:
                if not event.success:
                    raise ValueError(
                        f"{event.event_id}: fixed replay unexpectedly succeeded"
                    )
                returned = (
                    returned.model_dump(mode="json")
                    if isinstance(returned, BaseModel)
                    else returned
                )
                try:
                    saved = json.loads(event.result)
                except (TypeError, ValueError):
                    saved = event.result
                if returned != saved:
                    raise ValueError(f"{event.event_id}: fixed replay result mismatch")
            if event.db_effect is not (before != live.db.model_dump(mode="json")):
                raise ValueError(f"{event.event_id}: fixed replay DB effect mismatch")
        actual.append(live.db.model_dump(mode="json") == target)
    if actual != expected:
        raise ValueError(
            f"{record.trajectory_id}: stored official DB prefixes differ from pinned replay"
        )
    return len(actual)


def fixed_communicate_verifier(source):
    """Use the pinned predicate bodies without importing Tau2's LLM runner."""

    def tree(path):
        raw = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{PINNED_TAU2}:{path}"]
        )
        return ast.parse(raw.decode(), filename=path)

    message = tree("src/tau2/data_model/message.py")
    parent = next(
        n
        for n in message.body
        if isinstance(n, ast.ClassDef) and n.name == "ParticipantMessageBase"
    )
    text = next(
        n
        for n in parent.body
        if isinstance(n, ast.FunctionDef) and n.name == "has_text_content"
    )
    cls = ast.ClassDef(
        name="AssistantMessage",
        bases=[ast.Name(id="SimpleNamespace", ctx=ast.Load())],
        keywords=[],
        body=[text],
        decorator_list=[],
    )
    evaluator = tree("src/tau2/evaluator/evaluator_communicate.py")
    comm = next(
        n
        for n in evaluator.body
        if isinstance(n, ast.ClassDef) and n.name == "CommunicateEvaluator"
    )
    predicate = next(
        n
        for n in comm.body
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_communicate_info"
    )
    mod = ast.Module(body=[cls, predicate], type_ignores=[])
    ast.fix_missing_locations(mod)
    scope = {
        "SimpleNamespace": SimpleNamespace,
        "Message": object,
        "CommunicateCheck": lambda **kw: SimpleNamespace(**kw),
    }
    exec(compile(mod, "pinned_communicate_predicate", "exec"), scope)
    # At module scope the original @classmethod descriptor is bound explicitly.
    evaluate = scope["evaluate_communicate_info"].__func__

    def verify(record):
        required = (
            record.scoring_inputs["judge"]["task"]["evaluation_criteria"].get(
                "communicate_info"
            )
            or []
        )
        messages = [
            scope["AssistantMessage"](content=m.get("content"))
            if m.get("role") == "assistant"
            else SimpleNamespace()
            for m in record.environment_transcript
        ]
        names = [c["check_id"] for c in record.progress_trace["checks"]]
        for t, length in enumerate(record.progress_trace["prefix_lengths"]):
            native = [c.met for c in evaluate(None, messages[:length], required)]
            saved = [
                record.progress_trace["check_values"][t][names.index(f"comm:{i}")]
                for i in range(len(required))
            ]
            if saved != native:
                raise ValueError(
                    f"{record.trajectory_id}: official COMMUNICATE prefix mismatch"
                )
        return len(messages)

    return verify


def enabled_record(record, *, subgoals, initial):
    frozen = deepcopy(record.scoring_inputs)
    frozen["reward_project_config"]["credit"]["post_completion_credit"] = (
        POST_COMPLETION_VERSION
    )
    frozen["reward_project_config"]["credit"]["db_subgoal_credit"] = (
        "db_subgoal_local_v1" if subgoals is not None else None
    )
    if subgoals is not None:
        frozen["db_subgoal_inputs"] = {
            "initial_reservations": initial,
            "initial_state_fingerprint": record.progress_trace[
                "initial_state_fingerprint"
            ],
        }
    completion = completion_trace_from_scoring(
        frozen,
        record.tool_events,
        record.assistant_turns,
        record.custom_reward,
        subgoals,
    )
    updated = record.model_copy(
        update={
            "scoring_inputs": frozen,
            "subgoal_trace": subgoals,
            "completion_trace": completion,
            "metadata": record.metadata
            | {"scoring_inputs_sha256": sha256_json(frozen)},
        }
    )
    return credit_from_record(updated), completion


def audit(source, output, tau2_source):
    source, output = source.resolve(), output.resolve()
    if (
        output.exists()
        or source == output
        or source in output.parents
        or output in source.parents
    ):
        raise ValueError("completion audit needs a separate fresh output directory")
    paths = sorted((source / "group_audits").glob("*.json"))
    if not paths:
        raise ValueError("no original PPO groups")
    cls, db = load_fixed_tools(tau2_source)
    verify_comm = fixed_communicate_verifier(tau2_source)
    output.mkdir(parents=True)
    (output / "groups").mkdir()
    counts, skips, mistakes, doubles, seen, named = (
        Counter(),
        Counter(),
        [],
        [],
        set(),
        None,
    )
    for index, path in enumerate(paths):
        report = json.loads(path.read_text(encoding="utf-8"))
        records = []
        for row in report["inputs"]:
            rid = row["trajectory_id"]
            if (
                len(rid) != 32
                or any(c not in "0123456789abcdef" for c in rid)
                or rid in seen
            ):
                raise ValueError("unsafe or reused original trajectory identity")
            seen.add(rid)
            records.append(
                TrajectoryRecord.model_validate_json(
                    (source / "trajectories" / f"{rid}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        initial = {}

        def replay(record):
            result = replay_initial(record, cls, db)
            if result[2]:
                raise ValueError(result[2])
            initial[record.trajectory_id] = result[0]
            return result

        db_report = compare_group(report, records, replay)
        db_rows, post_rows, both_rows, traces = [], [], [], []
        for record, original, db_item in zip(
            records, report["inputs"], db_report["trajectories"], strict=True
        ):
            db_rows.append(original | {"subgoal_trace": db_item["subgoal_trace"]})
            post, trace = enabled_record(record, subgoals=None, initial=None)
            both, both_trace = enabled_record(
                record,
                subgoals=db_item["subgoal_trace"],
                initial=initial[record.trajectory_id],
            )
            post_rows.append(post)
            both_rows.append(both)
            traces.append((trace, both_trace))
            counts["trajectories"] += 1
            counts["regression_turns"] += len(trace["regression_turns"])
            counts["regression_trajectories"] += bool(trace["regression_turns"])
            counts["initially_complete_regression_trajectories"] += trace[
                "initial_success"
            ] and bool(trace["regression_turns"])
            counts["completion_bonus_trajectories"] += any(trace["turn_bonus"])
            counts["completion_floor_trajectories"] += (
                trace["positive_floor_turn"] is not None
            )
            counts["recovered_final_success_trajectories"] += (
                bool(trace["regression_turns"]) and record.official_scores.reward == 1
            )
            if trace["skip_reason"]:
                skips[trace["skip_reason"]] += 1
            if trace["regression_turns"]:
                counts["independently_replayed_prefixes"] += verify_db_prefixes(
                    record, cls, db
                )
                verify_comm(record)
            if (
                both_trace["db_positive_credit"]
                + both_trace["completion_positive_credit"]
                > 0.5 + 1e-12
            ):
                doubles.append(record.trajectory_id)
        configs = {
            "baseline": CreditConfig(**report["credit"]["config"]),
            "db_only": CreditConfig(
                **(
                    report["credit"]["config"]
                    | {"db_subgoal_credit": "db_subgoal_local_v1"}
                )
            ),
            "post_only": CreditConfig(
                **(
                    report["credit"]["config"]
                    | {"post_completion_credit": POST_COMPLETION_VERSION}
                )
            ),
            "combined": CreditConfig(
                **(
                    report["credit"]["config"]
                    | {
                        "db_subgoal_credit": "db_subgoal_local_v1",
                        "post_completion_credit": POST_COMPLETION_VERSION,
                    }
                )
            ),
        }
        modes = {
            name: compute_group_credit(rows, configs[name])
            for name, rows in {
                "baseline": report["inputs"],
                "db_only": db_rows,
                "post_only": post_rows,
                "combined": both_rows,
            }.items()
        }
        assert modes["baseline"] == report["credit"]
        for mode in modes.values():
            assert all(
                mode[k] == modes["baseline"][k]
                for k in ("score_mean", "score_std", "turn_mean")
            )
        details = []
        for j, (record, original, (trace, both_trace)) in enumerate(
            zip(records, report["inputs"], traces, strict=True)
        ):
            values = {
                name: mode["trajectories"][j]["token_advantages"]
                for name, mode in modes.items()
            }
            turns = {}
            policy = set(original["policy_credit"]["violating_turns"])
            for k, t in enumerate(original["response_turn_ids"]):
                if t < 0:
                    assert all(v[k] == 0 for v in values.values())
                    continue
                turns[t + 1] = {name: v[k] for name, v in values.items()}
            for t, vals in turns.items():
                for mode, prior, tr in (
                    ("post_only", "baseline", trace),
                    ("combined", "db_only", both_trace),
                ):
                    changed = abs(vals[mode] - vals[prior]) > 1e-12
                    regression = t in tr["regression_turns"]
                    completion = t == tr["positive_floor_turn"]
                    if changed and not regression and not completion:
                        mistakes.append(
                            dict(
                                trajectory_id=record.trajectory_id,
                                turn=t,
                                mode=mode,
                                reason="unrelated turn changed",
                            )
                        )
                    if completion:
                        floor = max(
                            tr["min_positive_advantage"],
                            tr["positive_margin"]
                            - original["process_credit"]["turn_costs"][t - 1],
                        )
                        assert vals[mode] >= floor > 0 and t - 1 not in policy
                    if regression:
                        assert (
                            vals[mode]
                            <= -1 - original["process_credit"]["turn_costs"][t - 1]
                        )
                        if t - 1 in policy:
                            assert vals[mode] == vals[prior]
                    if t - 1 in policy:
                        assert (
                            vals[mode]
                            <= -1 - original["process_credit"]["turn_costs"][t - 1]
                        )
            counts["regressions_already_policy_limited"] += len(
                policy & {t - 1 for t in trace["regression_turns"]}
            )
            if trace["positive_floor_turn"] is not None:
                counts["positive_final_completion_trajectories"] += (
                    turns[trace["positive_floor_turn"]]["post_only"] > 0
                )
            item = {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "official_reward": record.official_scores.reward,
                "train_reward": original["score"],
                "completion_trace": trace,
                "combined_completion_trace": both_trace,
                "token_advantage_hashes": {
                    n: sha256_json(v) for n, v in values.items()
                },
                "turn_advantages": turns,
                "completion_floor_adjustments": {
                    name: mode["trajectories"][j]["completion_turn_floor_adjustments"]
                    for name, mode in modes.items()
                    if name in {"post_only", "combined"}
                },
            }
            details.append(item)
            if record.trajectory_id == "5e8df95f6e75493cbe92a20afb65c714":
                named = item
                assert trace["first_success_turn"] == 6 and trace[
                    "regression_turns"
                ] == [10]
                assert math.isclose(turns[6]["post_only"], 0.5, abs_tol=1e-12)
                assert turns[6]["combined"] == turns[6]["post_only"]
                assert (
                    turns[10]["post_only"] <= -1 and record.official_scores.reward == 0
                )
        result = {
            "source_audit_fingerprint": report["audit_fingerprint"],
            "members": report["members"],
            "configs": {n: c.audit_config() for n, c in configs.items()},
            "trajectories": details,
        }
        (output / "groups" / path.name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts["groups"] += 1
        if (index + 1) % 12 == 0:
            print(f"audited {index + 1}/{len(paths)} groups", flush=True)
    summary = dict(counts) | {
        "skipped_tasks": dict(skips),
        "baseline_token_max_abs_error": 0.0,
        "double_reward_trajectories": doubles,
        "mispenalized_turns": mistakes,
        "task12_regression": named,
        "fixed_tau2_commit": PINNED_TAU2,
        "acceptance_passed": not mistakes and not doubles,
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
    summary = audit(args.source, args.output, args.tau2_source)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "task12_regression"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if not summary["acceptance_passed"]:
        raise SystemExit("post-completion offline acceptance failed")


if __name__ == "__main__":
    main()
