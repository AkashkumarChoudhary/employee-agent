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
    s = Settings(_env_file=None, provider="fake", api_keys="key-a,key-b",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    prov = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    return create_app(s, provider=prov)


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _files():
    return {"resume": ("resume.txt", b"Ada Lovelace. 5 years Python and Django.", "text/plain")}


async def test_create_job_pauses_for_human(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "Senior Python engineer", "role": "hr_analyst"},
                         files=_files())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "awaiting_human"
        assert body["job_id"]


async def test_get_job_returns_assessment(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-a"})
        assert r.status_code == 200
        assert r.json()["assessment"]["candidate_name"] == "Ada Lovelace"


async def test_bola_other_key_cannot_read(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-b"})
        assert r.status_code == 404


async def test_rejects_unsupported_file_type(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "x"},
                         files={"resume": ("resume.exe", b"nope", "application/octet-stream")})
        assert r.status_code == 415


async def test_missing_job_description_is_422(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"}, files=_files())
        assert r.status_code == 422
