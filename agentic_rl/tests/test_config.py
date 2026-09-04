from tau2_agentic_rl.config import expand_env


def test_env_placeholder_default(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_VALUE", raising=False)
    assert expand_env("${MISSING_VALUE:-fallback}") == "fallback"


def test_env_placeholder_override(monkeypatch) -> None:
    monkeypatch.setenv("SET_VALUE", "exact")
    assert expand_env("${SET_VALUE:-fallback}") == "exact"
