from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig
from tau2_agentic_rl.judge.prompts import rubric_fingerprint


def test_offline_rubric_fingerprint_changes_even_if_rule_id_is_reused():
    before = [{"criterion_id": "x", "description": "old rule"}]
    after = [{"criterion_id": "x", "description": "new rule"}]
    assert rubric_fingerprint([], before, {}) != rubric_fingerprint([], after, {})


def test_cache_key_changes_with_model_provider_or_rubric() -> None:
    messages = [{"role": "user", "content": "frozen"}]
    first = DeepSeekJudge(
        JudgeConfig(model="judge-a", cache_dir="outputs/test_judge_cache/a")
    )
    second = DeepSeekJudge(
        JudgeConfig(model="judge-b", cache_dir="outputs/test_judge_cache/b")
    )
    third = DeepSeekJudge(
        JudgeConfig(
            model="judge-a",
            rubric_version="v2",
            cache_dir="outputs/test_judge_cache/c",
        )
    )
    assert first.cache_identity(messages) != second.cache_identity(messages)
    assert first.cache_identity(messages) != third.cache_identity(messages)
