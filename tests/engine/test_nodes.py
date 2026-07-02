from employee_agent.engine import nodes
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, Chunk

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace",
    years_experience=5.0,
    top_skills=["python", "django"],
    skill_matches=[],
    overall_match_score=80,
    recommendation="advance",
    rationale="Strong match.",
)


def _base_state(**overrides):
    state = {
        "job_id": "eng-1",
        "role_config": get_role("hr_analyst"),
        "job_description": "Senior Python engineer with Django.",
        "parsed_resume": "Ada Lovelace. 5 years Python and Django.",
        "retrieved_chunks": [],
        "assessment": None,
        "verifier_verdict": None,
        "retry_count": 0,
        "status": "running",
    }
    state.update(overrides)
    return state


async def test_manager_sets_running():
    out = await nodes.make_manager()(_base_state(status="new"))
    assert out["status"] == "running"
    assert out["retry_count"] == 0


async def test_parser_normalizes_and_indexes():
    retr = Retriever(FakeProvider(), VectorStore())
    parser = nodes.make_parser(retr)
    out = await parser(_base_state(job_id="eng-parse",
                                   parsed_resume="  Ada   Lovelace\n\nPython  Django "))
    assert out["parsed_resume"] == "Ada Lovelace Python Django"
    # side effect: resume chunks were indexed and are now retrievable
    hits = await retr.retrieve("eng-parse", "Ada Lovelace Python Django", k=3)
    assert hits


async def test_retriever_node_sets_retrieved_chunks():
    retr = Retriever(FakeProvider(), VectorStore())
    await retr.index("eng-ret", [Chunk(text="python and django", source="resume", score=0.0)])
    out = await nodes.make_retriever_node(retr)(
        _base_state(job_id="eng-ret", job_description="python and django")
    )
    assert out["retrieved_chunks"]
    assert out["retrieved_chunks"][0].text == "python and django"


async def test_analyst_produces_structured_assessment():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await nodes.make_analyst(provider)(
        _base_state(retrieved_chunks=[Chunk(text="5 yrs Python", source="resume", score=0.9)])
    )
    assert out["assessment"] == CANNED
    assert "status" not in out
