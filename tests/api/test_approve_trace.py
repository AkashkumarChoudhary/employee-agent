import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


def _app(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    prov = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    return create_app(s, provider=prov)


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _files():
    return {"resume": ("resume.txt", b"Ada Lovelace. Python and Django.", "text/plain")}


async def _new_job(c):
    return (await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]


async def test_approve_finalizes_done(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.post(f"/jobs/{job_id}/approve", headers={"x-api-key": "key-a"},
                         json={"action": "approve"})
        assert r.status_code == 200
        assert r.json()["status"] == "done"
        assert r.json()["assessment"]["human_approved"] is True


async def test_reject_finalizes_error(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.post(f"/jobs/{job_id}/approve", headers={"x-api-key": "key-a"},
                         json={"action": "reject"})
        assert r.json()["status"] == "error"


async def test_trace_lists_execution_path(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.get(f"/jobs/{job_id}/trace", headers={"x-api-key": "key-a"})
        assert r.status_code == 200
        nodes = [s["node"] for s in r.json()["steps"]]
        for expected in ["manager", "parser", "retriever", "analyst", "verifier"]:
            assert expected in nodes
