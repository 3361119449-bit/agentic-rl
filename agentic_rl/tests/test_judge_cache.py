from tau2_agentic_rl.judge.client import DeepSeekJudge, JudgeConfig


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
