from employee_agent.config import Settings, get_settings


def test_defaults_apply_when_env_absent(monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("TEMPERATURE", raising=False)
    s = Settings(_env_file=None)
    assert s.provider == "gemini"
    assert s.gemini_model == "gemini-2.0-flash"
    assert s.temperature == 0.0
    assert s.enable_failover is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PROVIDER", "fake")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    s = Settings(_env_file=None)
    assert s.provider == "fake"
    assert s.temperature == 0.7


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
