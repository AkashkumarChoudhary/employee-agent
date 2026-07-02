from employee_agent.engine import nodes
from employee_agent.schemas import CandidateAssessment

ASSESSMENT = CandidateAssessment(
    candidate_name="Ada", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=70, recommendation="hold", rationale="ok",
)


async def test_gate_sets_awaiting_human():
    out = await nodes.make_gate()({"status": "running"})
    assert out["status"] == "awaiting_human"


async def test_finalizer_done_when_approved():
    approved = ASSESSMENT.model_copy(update={"human_approved": True})
    out = await nodes.make_finalizer()({"assessment": approved})
    assert out["status"] == "done"


async def test_finalizer_error_when_not_approved():
    out = await nodes.make_finalizer()({"assessment": ASSESSMENT})  # human_approved False
    assert out["status"] == "error"
