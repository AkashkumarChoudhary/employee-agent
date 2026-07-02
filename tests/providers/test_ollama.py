from unittest.mock import MagicMock

import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.ollama_provider import OllamaProvider
from employee_agent.schemas import VerifierVerdict


def _provider(monkeypatch, client):
    monkeypatch.setattr(
        "employee_agent.providers.ollama_provider.ollama.AsyncClient",
        lambda **kw: client,
    )
    return OllamaProvider(model="llama3.1", host="http://localhost:11434")


async def test_generate_structured_parses(monkeypatch):
    client = MagicMock()
    client.chat = _async_return(
        {"message": {"content":
         '{"grounded": false, "unsupported_claims": ["x"], "action": "retry_analysis"}'}}
    )
    p = _provider(monkeypatch, client)
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got.action == "retry_analysis"


async def test_error_becomes_provider_error(monkeypatch):
    client = MagicMock()
    client.chat = _async_raise(RuntimeError("conn refused"))
    p = _provider(monkeypatch, client)
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)


async def test_embed(monkeypatch):
    client = MagicMock()
    client.embed = _async_return({"embeddings": [[0.5, 0.6]]})
    p = _provider(monkeypatch, client)
    out = await p.embed(["a"])
    assert out == [[0.5, 0.6]]


def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f


def _async_raise(exc):
    async def _f(*a, **k):
        raise exc
    return _f
