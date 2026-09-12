import asyncio
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from test_judge_evidence import install_mock_api
from test_review_boundaries import scoring_failure_record
from test_rollout_integration import minimal_loop

from tau2_agentic_rl.evaluation import evaluation_coverage, initialize_evaluation
from tau2_agentic_rl.judge.client import JudgeConfig
from tau2_agentic_rl.schemas import OfficialScores
from tau2_agentic_rl.storage import TrajectoryStore
from tau2_agentic_rl.user_simulation import (
    FILTER_VERSION,
    SYSTEM_PROMPT,
    UserSimulationInputs,
    UserSimulationJudge,
    UserSimulationRejected,
    UserSimulationVerdict,
    check_user_simulation,
    make_user_sim_inputs,
    replacement_seed,
)
from tau2_agentic_rl.versions import sha256_json

SCENARIO = {
    "persona": None,
    "instructions": {
        "domain": "airline",
        "reason_for_call": "Cancel reservation A",
        "known_info": "reservation A",
        "unknown_info": "refund amount",
        "task_instructions": "If offered a refund, accept it.",
    },
}


def verdict(valid=True, kind="invented_fact"):
    return {
        "user_sim_valid": valid,
        "violation_type": "none" if valid else kind,
        "severity": "none" if valid else "hard",
        "turn_ids": [] if valid else [0],
        "reason": "No hard violation"
        if valid
        else "User invented a new reservation B.",
    }


def attach_valid_screen(row, valid=True):
    messages = row.setdefault(
        "environment_transcript",
        [
            {"role": "user", "content": "Cancel A", "turn_idx": 0},
            {"role": "assistant", "content": "Done", "turn_idx": 1},
        ],
    )
    for i, message in enumerate(messages):
        message["turn_idx"] = i
    inputs = make_user_sim_inputs(
        "Exact user guidelines", SCENARIO, messages
    ).model_dump()
    row["user_sim_inputs"] = inputs
    row["user_sim_result"] = verdict(valid)
    row.setdefault("metadata", {}).update(
        user_sim_filter_version=FILTER_VERSION,
        user_sim_inputs_sha256=sha256_json(inputs),
    )
    if row.get("scoring_inputs") is not None:
        row["scoring_inputs"]["judge"]["trajectory"]["messages"] = deepcopy(messages)
        row["metadata"]["scoring_inputs_sha256"] = sha256_json(row["scoring_inputs"])
    return row


def test_allowlist_removes_all_outcome_and_reference_fields():
    secret = "MUST_NOT_REACH_QUALITY_JUDGE"
    history = [
        {
            "role": "user",
            "turn_idx": 0,
            "content": "Cancel A",
            "raw_data": secret,
            "official_reward": secret,
            "expected_actions": secret,
        },
        {
            "role": "assistant",
            "turn_idx": 1,
            "tool_calls": [
                {
                    "id": "c",
                    "name": "lookup",
                    "arguments": {"id": "A"},
                    "raw_data": secret,
                }
            ],
        },
        {"role": "tool", "turn_idx": 2, "content": '{"id":"A"}', "expected_db": secret},
    ]
    inputs = make_user_sim_inputs("guidelines", SCENARIO, history)
    rendered = json.dumps(
        UserSimulationJudge.build_messages(inputs=inputs.model_dump())
    )
    assert secret not in rendered
    assert "lookup" in rendered and "Cancel A" in rendered
    with pytest.raises(ValidationError):
        UserSimulationInputs.model_validate(
            {**inputs.model_dump(), "official_reward": 0}
        )
    with pytest.raises(ValidationError):
        make_user_sim_inputs("g", {**SCENARIO, "expected_actions": secret}, history)


@pytest.mark.parametrize(
    "kind",
    [
        "invented_fact",
        "scenario_contradiction",
        "goal_drift",
        "withheld_known_information",
        "conditional_behavior_violation",
        "invalid_termination",
    ],
)
def test_six_hard_violations_are_supported(kind):
    result = UserSimulationVerdict.model_validate(verdict(False, kind))
    assert not result.user_sim_valid


@pytest.mark.parametrize(
    "changes",
    [
        {"user_sim_valid": "false"},
        {"turn_ids": [True]},
        {"turn_ids": [-1]},
        {"turn_ids": [0.0]},
        {"severity": "soft"},
        {"violation_type": "agent_failure"},
        {"turn_ids": []},
        {"reason": " "},
    ],
)
def test_invalid_filter_output_fails_closed(changes):
    with pytest.raises(ValidationError):
        UserSimulationVerdict.model_validate({**verdict(False), **changes})


def test_filter_references_must_point_to_user_replies():
    inputs = make_user_sim_inputs(
        "g",
        SCENARIO,
        [
            {"role": "user", "turn_idx": 0, "content": "hi"},
            {"role": "assistant", "turn_idx": 1, "content": "done"},
        ],
    )
    with pytest.raises(ValueError, match="non-user"):
        UserSimulationJudge._validate_requested_criteria(
            UserSimulationVerdict.model_validate({**verdict(False), "turn_ids": [1]}),
            {"inputs": inputs},
        )
    assert "user believes Agent's claim of completion" in SYSTEM_PROMPT
    assert "Agent failure or reward zero must NEVER" in SYSTEM_PROMPT


@pytest.mark.parametrize("reward", [0, 1])
@pytest.mark.parametrize("valid", [False, True])
def test_filter_is_outcome_blind_and_keeps_failed_agents(scratch_dir, reward, valid):
    record = scoring_failure_record()
    record.official_scores = OfficialScores(reward=reward)
    row = attach_valid_screen(record.model_dump())
    record = type(record).model_validate(row)
    captured = []

    class Judge:
        async def evaluate(self, **inputs):
            captured.append(inputs)
            return (
                UserSimulationVerdict.model_validate(verdict(valid)),
                "raw",
                "hash",
                "cache",
            )

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    assert asyncio.run(check_user_simulation(record, Judge(), store)) == valid
    assert set(captured[0]) == {"inputs"}
    assert set(captured[0]["inputs"]) == {
        "guidelines",
        "user_scenario",
        "conversation",
        "user_replies",
    }
    assert record.official_scores.reward == reward


def test_quality_client_validation_retry_and_cache_use_only_safe_inputs(
    scratch_dir, monkeypatch
):
    bad = {**verdict(False), "turn_ids": [99]}
    calls = install_mock_api(monkeypatch, [bad, verdict(True)])
    judge = UserSimulationJudge(
        JudgeConfig(
            model="quality-fixture",
            max_retries=1,
            cache_dir=str(scratch_dir),
            prompt_version=FILTER_VERSION,
            rubric_version=FILTER_VERSION,
        )
    )
    inputs = make_user_sim_inputs(
        "g", SCENARIO, [{"role": "user", "turn_idx": 0, "content": "hi"}]
    ).model_dump()
    result = asyncio.run(judge.evaluate(inputs=inputs))
    assert result[0].user_sim_valid and len(calls) == 2
    assert asyncio.run(judge.evaluate(inputs=inputs)) == result
    assert len(calls) == 2
    assert json.loads(calls[0]["messages"][1]["content"].split("\n", 1)[1]) == inputs


def test_check_api_failure_stays_pending_and_does_not_resample(scratch_dir):
    record = scoring_failure_record()
    record = type(record).model_validate(attach_valid_screen(record.model_dump()))

    class Judge:
        async def evaluate(self, **kwargs):
            raise RuntimeError("API unavailable")

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    with pytest.raises(RuntimeError, match="API unavailable"):
        asyncio.run(check_user_simulation(record, Judge(), store))
    saved = next(store.records())
    assert saved.user_sim_result is None
    assert saved.metadata["failure_phase"] == "user_sim_judge"
    assert saved.environment_seed == record.environment_seed


@pytest.mark.parametrize("evaluation", [False, True])
def test_actor_only_replaces_rejected_users_and_changes_seed(
    scratch_dir, monkeypatch, evaluation
):
    loop, _ = minimal_loop(scratch_dir)
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 2}
    if evaluation:
        monkeypatch.setenv("EVALUATION_MANIFEST_ID", "test")
    else:
        monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    calls = []

    async def generate(params, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UserSimulationRejected("bad user")
        return "accepted tokens"

    loop._run_trajectory = generate
    if evaluation:
        with pytest.raises(UserSimulationRejected):
            asyncio.run(
                loop._run_valid_trajectory(
                    {}, trajectory_id="abcdef00", extra_info={"environment_seed": 42}
                )
            )
        assert len(calls) == 1  # Driver owns frozen eval slot refills.
    else:
        assert (
            asyncio.run(
                loop._run_valid_trajectory(
                    {}, trajectory_id="abcdef00", extra_info={"environment_seed": 42}
                )
            )
            == "accepted tokens"
        )
        assert calls[0]["trajectory_id"] != calls[1]["trajectory_id"]
        assert (
            calls[0]["extra_info"]["environment_seed"]
            != calls[1]["extra_info"]["environment_seed"]
        )

    async def broken(*args, **kwargs):
        raise RuntimeError("quality judge API error")

    loop._run_trajectory = broken
    with pytest.raises(RuntimeError, match="API error"):
        asyncio.run(
            loop._run_valid_trajectory(
                {}, trajectory_id="abcdef00", extra_info={"environment_seed": 42}
            )
        )


def test_official_only_coverage_excludes_only_invalid_users_and_keeps_zero_reward(
    scratch_dir,
):
    from scripts.summarize_evaluation import summarize

    root = scratch_dir / "eval"
    manifest = initialize_evaluation(
        root,
        {
            "task_ids": ["0"],
            "samples_per_task": 4,
            "split": "internal_dev",
            "record_split": "internal_dev",
            "reward_judge_enabled": False,
            "user_sim_filter_enabled": True,
        },
        resume=False,
    )
    store = TrajectoryStore(root / "trajectories", attach_evaluation_identity=False)
    records = []
    for slot in range(4):
        record = scoring_failure_record()
        record.trajectory_id = str(slot)
        record.official_scores = OfficialScores(reward=0)
        record.metadata.update(
            evaluation_manifest_id=manifest["manifest_id"],
            evaluation_sample_index=slot,
            failure_phase=None,
            reward_judge_enabled=False,
        )
        record = type(record).model_validate(
            attach_valid_screen(record.model_dump(), valid=slot != 0)
        )
        store.save(record)
        records.append(record)
    coverage = evaluation_coverage(store.root, manifest)
    assert coverage["valid_samples"] == 3
    assert coverage["missing_slots"] == [
        {"task_id": "0", "sample_index": 0, "user_sim_attempt": 1}
    ]
    record = records[0].model_copy(deep=True)
    record.trajectory_id = "replacement"
    record.user_sim_result = verdict(True)
    record.metadata["user_sim_attempt"] = 1
    record.environment_seed = replacement_seed(record.environment_seed, 1)
    store.save(record)
    report = summarize(store.root)
    assert report["aggregate"] == {"official_pass1": 0, "official_pass4": 0}
    assert report["valid_samples"] == 4 and report["user_sim_rejections"] == 1
    # Check failure must hold this exact interaction, never open a refill slot.
    record.user_sim_result = None
    record.metadata["failure_phase"] = "user_sim_judge"
    store.save(record)
    coverage = evaluation_coverage(store.root, manifest)
    assert (
        not coverage["missing_slots"] and len(coverage["user_sim_pending_records"]) == 1
    )


def test_offline_checks_all_records_and_preserves_source_outcomes(scratch_dir):
    from scripts.screen_user_simulations import screen_records

    records = []
    for i in range(3):
        record = scoring_failure_record()
        record.trajectory_id = str(i)
        record = type(record).model_validate(attach_valid_screen(record.model_dump()))
        records.append(record)
    before = [deepcopy(r.official_scores) for r in records]
    responses = iter([True, False, None])

    class Judge:
        async def evaluate(self, **kwargs):
            response = next(responses)
            if response is None:
                raise RuntimeError("API failure")
            return (
                UserSimulationVerdict.model_validate(verdict(response)),
                "r",
                "h",
                "k",
            )

    report = asyncio.run(screen_records(records, Judge(), scratch_dir))
    assert report["counts"] == {"accepted": 1, "rejected": 1, "pending": 1}
    assert [r.official_scores for r in records] == before


@pytest.mark.parametrize(
    "failure_phase", [None, "judge", "reward_scoring", "user_sim_judge"]
)
def test_missing_quality_verdict_cannot_bypass_filter(scratch_dir, failure_phase):
    manifest = initialize_evaluation(
        scratch_dir / "eval",
        {
            "task_ids": ["0"],
            "samples_per_task": 4,
            "split": "internal_dev",
            "record_split": "internal_dev",
            "reward_judge_enabled": False,
            "user_sim_filter_enabled": True,
        },
        resume=False,
    )
    store = TrajectoryStore(
        scratch_dir / "eval/trajectories", attach_evaluation_identity=False
    )
    record = scoring_failure_record()
    record = type(record).model_validate(attach_valid_screen(record.model_dump()))
    record.metadata.update(
        evaluation_manifest_id=manifest["manifest_id"], failure_phase=failure_phase
    )
    record.user_sim_result = None
    store.save(record)
    coverage = evaluation_coverage(store.root, manifest)
    assert coverage["valid_samples"] == 0
    assert len(coverage["user_sim_pending_records"]) == 1
    assert {s["sample_index"] for s in coverage["missing_slots"]} == {1, 2, 3}
    record.user_sim_inputs = None
    store.save(record)
    with pytest.raises(ValueError, match="lacks required"):
        evaluation_coverage(store.root, manifest)


def test_pending_record_is_saved_before_quality_api_request(scratch_dir):
    record = scoring_failure_record()
    record = type(record).model_validate(attach_valid_screen(record.model_dump()))
    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)

    class Judge:
        async def evaluate(self, **kwargs):
            saved = next(store.records())
            assert saved.user_sim_result is None
            assert saved.metadata["failure_phase"] == "user_sim_judge"
            raise asyncio.CancelledError("process shutdown")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(check_user_simulation(record, Judge(), store))
    assert next(store.records()).user_sim_result is None


def test_training_replacement_limit_never_returns_invalid_tokens(
    scratch_dir, monkeypatch
):
    loop, _ = minimal_loop(scratch_dir)
    loop.project["user_sim_filter"] = {"enabled": True, "max_resamples": 2}
    monkeypatch.delenv("EVALUATION_MANIFEST_ID", raising=False)
    calls = []

    async def generate(*args, **kwargs):
        calls.append(kwargs)
        raise UserSimulationRejected("invalid")

    loop._run_trajectory = generate
    with pytest.raises(UserSimulationRejected):
        asyncio.run(
            loop._run_valid_trajectory(
                {}, trajectory_id="abcdef00", extra_info={"environment_seed": 42}
            )
        )
    assert len(calls) == 3
    assert len({c["trajectory_id"] for c in calls}) == 3
    assert len({c["extra_info"]["environment_seed"] for c in calls}) == 3


@pytest.mark.parametrize("tamper", ["hash", "transcript"])
def test_corrupt_frozen_inputs_are_pending_without_any_api_call(scratch_dir, tamper):
    record = scoring_failure_record()
    record = type(record).model_validate(attach_valid_screen(record.model_dump()))
    if tamper == "hash":
        record.metadata["user_sim_inputs_sha256"] = "wrong"
    else:
        record.environment_transcript[0]["content"] = "changed after check"

    class Judge:
        async def evaluate(self, **kwargs):
            pytest.fail("corrupt evidence must never reach API")

    store = TrajectoryStore(scratch_dir, attach_evaluation_identity=False)
    with pytest.raises(ValueError):
        asyncio.run(check_user_simulation(record, Judge(), store))
    saved = next(store.records())
    assert saved.user_sim_result is None
    assert saved.metadata["failure_phase"] == "user_sim_judge"


def test_offline_dry_run_verifies_saved_inputs_without_writing_or_api(
    scratch_dir, monkeypatch, capsys
):
    import sys

    from scripts import screen_user_simulations

    source, output = scratch_dir / "source", scratch_dir / "output"
    store = TrajectoryStore(source, attach_evaluation_identity=False)
    record = scoring_failure_record()
    record = type(record).model_validate(attach_valid_screen(record.model_dump()))
    store.save(record)
    monkeypatch.setattr(
        sys,
        "argv",
        ["screen", str(source), str(output), "--config", "unused.yaml", "--dry-run"],
    )
    monkeypatch.setattr(
        screen_user_simulations,
        "build_user_sim_judge",
        lambda *a: pytest.fail("dry run must not construct client"),
    )
    screen_user_simulations.main()
    assert json.loads(capsys.readouterr().out)["api_calls"] == 0
    assert not output.exists()
    record.metadata["user_sim_inputs_sha256"] = "wrong"
    store.save(record)
    with pytest.raises(ValueError, match="hash mismatch"):
        screen_user_simulations.main()
    assert not output.exists()
