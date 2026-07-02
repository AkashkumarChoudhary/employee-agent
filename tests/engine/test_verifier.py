from employee_agent.engine import nodes
from employee_agent.engine.routing import MAX_RETRIES, route_after_verifier
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, Chunk, VerifierVerdict

ASSESSMENT = CandidateAssessment(
    candidate_name="Ada", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=80, recommendation="advance", rationale="ok",
)


def _state(**overrides):
    s = {
        "job_id": "v1", "role_config": None, "job_description": "jd",
        "parsed_resume": "r", "retrieved_chunks": [Chunk(text="python", source="resume", score=0.9)],
        "assessment": ASSESSMENT, "verifier_verdict": None, "retry_count": 0,
        "status": "running",
    }
    s.update(overrides)
    return s


async def test_verifier_accept_does_not_bump_retry():
    v = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    out = await nodes.make_verifier(FakeProvider(responses={VerifierVerdict: v}))(_state())
    assert out["verifier_verdict"].action == "accept"
    assert "retry_count" not in out  # unchanged


async def test_verifier_nonaccept_bumps_retry():
    v = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis")
    out = await nodes.make_verifier(FakeProvider(responses={VerifierVerdict: v}))(
        _state(retry_count=1)
    )
    assert out["retry_count"] == 2


def test_route_accept_goes_to_gate():
    s = _state(verifier_verdict=VerifierVerdict(grounded=True, unsupported_claims=[], action="accept"))
    assert route_after_verifier(s) == "gate"


def test_route_retry_retrieval_under_cap_goes_to_retriever():
    s = _state(retry_count=1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_retrieval"))
    assert route_after_verifier(s) == "retriever"


def test_route_retry_analysis_under_cap_goes_to_analyst():
    s = _state(retry_count=1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis"))
    assert route_after_verifier(s) == "analyst"


def test_route_over_cap_escalates_to_gate():
    s = _state(retry_count=MAX_RETRIES + 1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis"))
    assert route_after_verifier(s) == "gate"
