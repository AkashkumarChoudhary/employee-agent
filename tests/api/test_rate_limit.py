import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada", years_experience=1.0, top_skills=["p"], skill_matches=[],
    overall_match_score=50, recommendation="hold", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


async def test_rate_limit_returns_429(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="rl-key",
                 rate_limit="3/minute",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    app = create_app(s, provider=FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT}))
    files = {"resume": ("resume.txt", b"Ada. Python.", "text/plain")}
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        codes = []
        for _ in range(4):
            r = await c.post("/jobs", headers={"x-api-key": "rl-key"},
                             data={"job_description": "x"}, files=files)
            codes.append(r.status_code)
        assert codes[:3] == [200, 200, 200]
        assert codes[3] == 429
