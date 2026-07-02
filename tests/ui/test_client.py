import json

import httpx
import pytest

from employee_agent.ui.client import EmployeeAgentClient


def _client(handler, api_key="k"):
    return EmployeeAgentClient(
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"),
    )


def test_create_job_posts_multipart_with_key():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        seen["key"] = req.headers.get("x-api-key")
        seen["ct"] = req.headers.get("content-type", "")
        return httpx.Response(200, json={"job_id": "j1", "status": "awaiting_human"})

    out = _client(handler).create_job(
        job_description="jd", role="hr_analyst", filename="r.txt", content=b"resume"
    )
    assert out == {"job_id": "j1", "status": "awaiting_human"}
    assert seen["path"] == "/jobs"
    assert seen["key"] == "k"
    assert "multipart/form-data" in seen["ct"]


def test_approve_sends_action_json():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"job_id": "j1", "status": "done",
                                         "assessment": {"human_approved": True}})

    out = _client(handler).approve("j1", "approve")
    assert out["status"] == "done"
    assert captured["body"] == {"action": "approve", "edits": {}}


def test_get_job_and_trace():
    def handler(req):
        if req.url.path.endswith("/trace"):
            return httpx.Response(200, json={"job_id": "j1", "steps": [{"step": 0, "node": "manager"}]})
        return httpx.Response(200, json={"job_id": "j1", "status": "awaiting_human", "assessment": None})

    c = _client(handler)
    assert c.get_job("j1")["status"] == "awaiting_human"
    assert c.trace("j1")[0]["node"] == "manager"


def test_raises_on_http_error():
    def handler(req):
        return httpx.Response(401, json={"detail": "nope"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).get_job("j1")
