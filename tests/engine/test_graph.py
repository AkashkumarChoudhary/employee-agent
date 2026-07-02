from langgraph.checkpoint.memory import MemorySaver

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace",
    years_experience=5.0,
    top_skills=["python"],
    skill_matches=[],
    overall_match_score=88,
    recommendation="advance",
    rationale="Strong Python and Django background.",
)


def _wire():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    retriever = Retriever(provider, VectorStore())
    graph = build_graph(provider, retriever, checkpointer=MemorySaver())
    return graph


def test_new_state_is_fully_populated():
    s = new_state("j1", get_role("hr_analyst"), "JD text", "resume text")
    assert s["status"] == "running"
    assert s["retrieved_chunks"] == []
    assert s["assessment"] is None
    assert s["retry_count"] == 0


async def test_graph_runs_end_to_end_and_produces_assessment():
    graph = _wire()
    state = new_state("graph-1", get_role("hr_analyst"),
                      "Senior Python engineer with Django.",
                      "Ada Lovelace. 5 years Python and Django.")
    final = await graph.ainvoke(state, {"configurable": {"thread_id": "graph-1"}})
    assert final["status"] == "done"
    assert final["assessment"] == CANNED
    assert final["retrieved_chunks"]  # resume was indexed then retrieved
