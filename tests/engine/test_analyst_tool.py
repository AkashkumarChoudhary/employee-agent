from employee_agent.engine.nodes import make_analyst
from employee_agent.mcp_tools.client import MCPToolClient
from employee_agent.providers.fake import FakeProvider
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, Chunk

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=80, recommendation="advance", rationale="Base.",
)


def _state(**overrides):
    s = {
        "job_id": "tool-1", "role_config": get_role("hr_analyst"),
        "job_description": "jd", "parsed_resume": "r",
        "retrieved_chunks": [Chunk(text="python", source="resume", score=0.9)],
        "assessment": None, "verifier_verdict": None, "retry_count": 0, "status": "running",
    }
    s.update(overrides)
    return s


class _BoomClient:
    async def call(self, tool, args):
        raise RuntimeError("mcp down")


async def test_analyst_appends_mcp_verification_when_allowlisted():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await make_analyst(provider, tool_client=client)(_state())
    assert "verify_certification" in out["assessment"].rationale
    assert "verified=True" in out["assessment"].rationale  # "python" is a known cert


async def test_analyst_without_tool_client_is_unchanged():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await make_analyst(provider)(_state())
    assert out["assessment"].rationale == "Base."


async def test_analyst_survives_tool_failure():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await make_analyst(provider, tool_client=_BoomClient())(_state())
    assert out["assessment"].candidate_name == "Ada Lovelace"  # still produced
    assert out["assessment"].rationale == "Base."  # no note appended on failure
