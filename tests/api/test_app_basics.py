import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider


def _app(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a,key-b",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    return create_app(s, provider=FakeProvider())


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_open(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


async def test_whoami_requires_key(tmp_path):
    async with _client(_app(tmp_path)) as c:
        assert (await c.get("/whoami")).status_code == 401
        assert (await c.get("/whoami", headers={"x-api-key": "bad"})).status_code == 401
        ok = await c.get("/whoami", headers={"x-api-key": "key-a"})
        assert ok.status_code == 200
        assert ok.json()["api_key"] == "key-a"
