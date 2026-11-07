import json
from copy import deepcopy

from test_simple_turn_target import frozen

from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.simple_turn import PROMPT, build_messages
from tau2_agentic_rl.reward.simple_turn import turn_request, validate_target_request
from tau2_agentic_rl.versions import sha256_json


def request():
    inputs = frozen()
    inputs["messages"][0]["content"] = (
        "Preserve this additional instruction.\n<policy>\n"
        + inputs["policy"]
        + "\n</policy>\nPreserve this trailing instruction."
    )
    return turn_request(inputs, 1, [])


def test_bound_policy_sent_once_without_changing_request_or_target():
    original = request()
    before = deepcopy(original)
    messages = build_messages(**original)
    sent = json.loads(messages[1]["content"])
    assert sent["policy"] == original["policy"]
    assert messages[1]["content"].count(original["policy"]) == 1
    system = sent["history"][0]["message"]["content"]
    assert system.startswith("Preserve this additional instruction.\n<policy>\n")
    assert system.endswith("\n</policy>\nPreserve this trailing instruction.")
    for field in original:
        if field != "history":
            assert sent[field] == original[field]
    assert sent["history"][1:] == original["history"][1:]
    assert original == before
    assert validate_target_request(sent) == original["target_message_index"]


def test_all_exact_duplicates_in_system_history_refer_to_one_authoritative_policy():
    original = request()
    original["history"][0]["message"]["content"] += "\n" + original["policy"]
    sent = json.loads(build_messages(**original)[1]["content"])
    assert sent["policy"] not in sent["history"][0]["message"]["content"]
    assert sent["policy"] == original["policy"]


def test_user_tool_and_assistant_policy_quotes_are_preserved():
    original = request()
    for index in [1, 2, 3]:
        original["history"][index]["message"]["content"] = original["policy"]
    original["history"][3]["message"]["role"] = "tool"
    sent = json.loads(build_messages(**original)[1]["content"])
    assert sent["history"][1:] == original["history"][1:]


def test_nonmatching_system_rules_are_never_stripped():
    original = request()
    content = "Different authoritative system text: use supplied facts only."
    original["history"][0]["message"]["content"] = content
    sent = json.loads(build_messages(**original)[1]["content"])
    assert sent == original


def test_current_raw_generation_and_target_hash_are_not_deduplicated():
    inputs = frozen()
    inputs["messages"][4]["raw_generated_text"] = inputs["policy"]
    original = turn_request(inputs, 1, [])
    sent = json.loads(build_messages(**original)[1]["content"])
    assert sent["current_generation"] == original["current_generation"]
    assert sent["target_generation_text"] == original["target_generation_text"]
    assert sent["target_generation_sha256"] == original["target_generation_sha256"]
    assert validate_target_request(sent) == original["target_message_index"]


def test_changed_wire_view_has_a_distinct_cache_identity(scratch_dir):
    original = request()
    before = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": json.dumps(original, ensure_ascii=False, sort_keys=True),
        },
    ]
    after = build_messages(**original)
    client = DeepSeekJudge(JudgeConfig(model="fixture", cache_dir=str(scratch_dir)))
    assert sha256_json(client.cache_identity(before)) != sha256_json(
        client.cache_identity(after)
    )
