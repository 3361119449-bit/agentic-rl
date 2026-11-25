"""Decide the training reward branch across the eight sessions of one veRL prompt."""

import asyncio
from contextvars import ContextVar

from tau2_agentic_rl.reward.progress import validate_progress_inputs, validate_progress_trace
from tau2_agentic_rl.schemas import RewardResult
from tau2_agentic_rl.versions import sha256_json


OPTION = "full_official_group_terminal_only"
_session = ContextVar("official_group_session", default=None)


class OfficialGroup:
    def __init__(self, uid):
        self.uid = uid
        self.ready = asyncio.Event()
        self.members = {}
        self.decision = None

    async def session(self, slot, operation):
        token = _session.set((self, slot))
        try:
            return await operation
        finally:
            # Exhausted, rejected and cancelled sessions must release their siblings.
            if slot not in self.members:
                self.report(slot, None)
            _session.reset(token)

    def report(self, slot, trajectory_id):
        if slot in self.members:
            raise ValueError("official group session reported twice")
        self.members[slot] = trajectory_id
        if self.ready.is_set():
            return
        if trajectory_id is None:
            self.ready.set()
        elif len(self.members) == 8:
            ids = [self.members[i] for i in range(8)]
            if len(set(ids)) != 8:
                self.ready.set()
                raise ValueError("duplicate trajectory in official group")
            self.decision = {"uid": self.uid, "members": ids}
            self.ready.set()


def cancel_group_wait():
    current = _session.get()
    if current:
        current[0].ready.set()


async def choose_group_branch(official_reward, trajectory_id, lease=None):
    current = _session.get()
    if current is None:
        return None
    group, slot = current
    if lease is not None:
        lease.release()
    group.report(slot, trajectory_id if official_reward == 1 else None)
    await group.ready.wait()
    return group.decision


def terminal_only_reward(trace, turns):
    progress = validate_progress_trace(trace, turns=turns)[-1]
    return RewardResult(
        branch="normal", reward_mode="turn_local_v1", train_reward=1 + 0.5 * progress,
        strict_success=1.0, progress=progress, policy_gate=True, process_penalty=0.0,
        details={
            "scoring_branch": OPTION, "official_reward": 1.0, "task_completion": 1.0,
            "credit_version": "procredit-turn-v4", "checkset_fingerprint": trace["checkset_fingerprint"],
            "truncation_multiplier": 1.0, "policy_gate_applied": False,
        },
    )


def terminal_only_row(record, mapping):
    frozen = record.scoring_inputs
    group = frozen["official_group"]
    members = group.get("members") if isinstance(group, dict) else None
    if (not isinstance(members, list) or len(members) != 8
            or not all(isinstance(item, str) and item for item in members)
            or len(set(members)) != 8 or record.trajectory_id not in members or not group.get("uid")):
        raise ValueError("invalid frozen full-official group membership")
    metadata = record.metadata
    trajectory = frozen["judge"]["trajectory"]
    trace = record.progress_trace
    if (not frozen["reward_project_config"]["reward"].get(OPTION)
            or record.split != "train" or record.schema_version != "2.0"
            or record.official_scores is None or record.official_scores.reward != 1
            or metadata.get("failure_phase") or metadata.get("truncation_zero_applied")
            or metadata.get("judge_routing") != {"call": False, "reason": OPTION}
            or record.judge_result is not None or metadata.get("simple_turn_labels") is not None
            or any(metadata.get(k) is not None for k in ("judge_raw", "judge_prompt_hash", "judge_cache_key"))
            or trajectory["termination_reason"] != record.termination_reason
            or trajectory["messages"] != record.environment_transcript
            or trajectory["tool_events"] != [event.model_dump() for event in record.tool_events]
            or frozen["official_scores"] != record.official_scores.model_dump()
            or not trace or trace.get("version") != "progress-v3"
            or frozen.get("progress_trace") != trace
            or metadata.get("scoring_inputs_sha256") != sha256_json(frozen)):
        raise ValueError("invalid frozen full-official group evidence")
    validate_progress_inputs(trace, required_actions=frozen["required_actions"],
                             dependencies=frozen["action_dependencies"], transfer_rule=frozen["judge"]["transfer_rule"])
    reward = terminal_only_reward(trace, record.assistant_turns)
    if record.custom_reward != reward:
        raise ValueError("terminal group score differs from deterministic checks")
    return dict(official_group=group, trajectory_id=record.trajectory_id, task_id=record.task_id,
                policy_version=record.policy_version, credit_version="procredit-turn-v4",
                checkset_fingerprint=trace["checkset_fingerprint"], initial_state_fingerprint=trace["initial_state_fingerprint"],
                score=reward.train_reward, valid=True, terminal_success=1., final_progress=reward.progress,
                multiplier=1., phi=trace["phi"], response_turn_ids=mapping)
