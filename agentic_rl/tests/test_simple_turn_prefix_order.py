import json
from copy import deepcopy
from os.path import commonprefix

from test_simple_turn_target import frozen

from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.simple_turn import build_messages
from tau2_agentic_rl.reward.simple_turn import turn_request, validate_target_request
from tau2_agentic_rl.versions import sha256_json


def test_stable_policy_history_precede_target_with_identical_data():
    request = turn_request(frozen(), 1, [])
    before = deepcopy(request)
    content = build_messages(**request)[1]["content"]
    sent = json.loads(content)
    assert list(sent)[:3] == ["request_version", "policy", "history"]
    assert content.index('"history":') < content.index('"current_generation":')
    expected = deepcopy(request)
    expected["history"][0]["message"]["content"] = (
        "[See request.policy for the bound policy.]"
    )
    assert sent == expected
    assert request == before
    assert validate_target_request(sent) == request["target_message_index"]


def test_growing_history_shares_the_existing_history_prefix():
    inputs = frozen()
    first = build_messages(**turn_request(inputs, 0, []))[1]["content"]
    second = build_messages(**turn_request(inputs, 1, []))[1]["content"]
    history = json.dumps(
        json.loads(first)["history"], ensure_ascii=False, sort_keys=True
    )
    expected = first.index(history) + len(history) - 1
    assert len(commonprefix([first, second])) >= expected
    assert "hidden future marker" not in first + second


def test_insertion_order_keeps_wire_bytes_and_cache_identity_stable(scratch_dir):
    request = turn_request(frozen(), 1, [])

    def reverse_objects(value):
        if isinstance(value, dict):
            return {k: reverse_objects(v) for k, v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [reverse_objects(v) for v in value]
        return value

    a = build_messages(**request)
    b = build_messages(**reverse_objects(request))
    assert a == b
    client = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    assert client.cache_identity(a) == client.cache_identity(b)


def test_order_change_gets_distinct_key_from_old_alphabetical_wire(scratch_dir):
    after = build_messages(**turn_request(frozen(), 1, []))
    before = deepcopy(after)
    before[1]["content"] = json.dumps(
        json.loads(after[1]["content"]), ensure_ascii=False, sort_keys=True
    )
    client = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    assert json.loads(after[1]["content"]) == json.loads(before[1]["content"])
    assert sha256_json(client.cache_identity(after)) != sha256_json(
        client.cache_identity(before)
    )
