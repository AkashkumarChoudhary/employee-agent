from unittest.mock import MagicMock

import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.gemini import GeminiProvider
from employee_agent.schemas import VerifierVerdict


def _provider_with_mock_client(monkeypatch, client):
    monkeypatch.setattr(
        "employee_agent.providers.gemini.genai.Client",
        lambda **kw: client,
    )
    return GeminiProvider(api_key="k", model="gemini-2.0-flash",
                          embedding_model="text-embedding-004")


async def test_generate_structured_parses_json(monkeypatch):
    client = MagicMock()
    resp = MagicMock()
    resp.text = ('{"grounded": true, "unsupported_claims": [], "action": "accept"}')
    client.aio.models.generate_content = _async_return(resp)
    p = _provider_with_mock_client(monkeypatch, client)
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert isinstance(got, VerifierVerdict) and got.grounded is True


async def test_sdk_error_becomes_provider_error(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = _async_raise(RuntimeError("429"))
    p = _provider_with_mock_client(monkeypatch, client)
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)


async def test_embed_maps_values(monkeypatch):
    client = MagicMock()
    emb = MagicMock()
    emb.embeddings = [MagicMock(values=[0.1, 0.2]), MagicMock(values=[0.3, 0.4])]
    client.aio.models.embed_content = _async_return(emb)
    p = _provider_with_mock_client(monkeypatch, client)
    out = await p.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


# --- async helpers ---
def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f


def _async_raise(exc):
    async def _f(*a, **k):
        raise exc
    return _f
