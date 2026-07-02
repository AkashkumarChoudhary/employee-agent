from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace",
    years_experience=5.0,
    top_skills=["python"],
    skill_matches=[],
    overall_match_score=88,
    recommendation="advance",
    rationale="Strong Python and Django background.",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


def _wire():
    provider = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    retriever = Retriever(provider, VectorStore())
    graph = build_graph(provider, retriever, checkpointer=MemorySaver())
    return graph


def test_new_state_is_fully_populated():
    s = new_state("j1", get_role("hr_analyst"), "JD text", "resume text")
    assert s["status"] == "running"
    assert s["retrieved_chunks"] == []
    assert s["assessment"] is None
    assert s["retry_count"] == 0


async def test_graph_runs_then_pauses_for_human_and_finalizes():
    graph = _wire()
    cfg = {"configurable": {"thread_id": "graph-1"}}
    res = await graph.ainvoke(
        new_state("graph-1", get_role("hr_analyst"),
                  "Senior Python engineer with Django.",
                  "Ada Lovelace. 5 years Python and Django."),
        cfg,
    )
    assert "__interrupt__" in res  # pauses at the human gate
    assert (await graph.aget_state(cfg)).values["retrieved_chunks"]
    final = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert final["status"] == "done"
    assert final["assessment"] == CANNED.model_copy(update={"human_approved": True})
