import pytest

from employee_agent.config import Settings
from employee_agent.providers.factory import build_provider
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider


def test_fake_provider_selected():
    p = build_provider(Settings(_env_file=None, provider="fake"))
    assert isinstance(p, FakeProvider)


def test_gemini_without_key_raises():
    s = Settings(_env_file=None, provider="gemini", gemini_api_key=None)
    with pytest.raises(ValueError):
        build_provider(s)


def test_gemini_with_failover_wraps(monkeypatch):
    monkeypatch.setattr("employee_agent.providers.factory.GeminiProvider",
                        lambda **kw: FakeProvider())
    monkeypatch.setattr("employee_agent.providers.factory.OllamaProvider",
                        lambda **kw: FakeProvider())
    s = Settings(_env_file=None, provider="gemini", gemini_api_key="k",
                 enable_failover=True)
    p = build_provider(s)
    assert isinstance(p, FailoverProvider)
