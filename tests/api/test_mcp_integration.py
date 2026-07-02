import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="Base.",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


async def test_assessment_includes_mcp_verification(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a",
                 sqlite_path=str(tmp_path / "ck.sqlite"), chroma_path=str(tmp_path / "chroma"))
    app = create_app(s, provider=FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT}))
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"},
                               files={"resume": ("r.txt", b"Ada Lovelace. Python.", "text/plain")})).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-a"})
        assert "verify_certification" in r.json()["assessment"]["rationale"]
