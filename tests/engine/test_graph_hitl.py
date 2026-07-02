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
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
RETRY = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis")


def _wire(verdict):
    provider = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: verdict})
    return build_graph(provider, Retriever(provider, VectorStore()), checkpointer=MemorySaver())


def _state(job_id):
    return new_state(job_id, get_role("hr_analyst"),
                     "Senior Python engineer.", "Ada Lovelace. Python, Django.")


async def test_accept_pauses_for_human_then_finalizes_on_approve():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-approve"}}
    res = await graph.ainvoke(_state("hitl-approve"), cfg)
    assert "__interrupt__" in res
    snap = await graph.aget_state(cfg)
    assert snap.values["status"] == "awaiting_human"
    assert snap.next == ("hitl",)
    final = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert final["status"] == "done"
    assert final["assessment"].human_approved is True


async def test_reject_finalizes_as_error():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-reject"}}
    await graph.ainvoke(_state("hitl-reject"), cfg)
    final = await graph.ainvoke(Command(resume={"action": "reject"}), cfg)
    assert final["status"] == "error"
    assert final["assessment"].human_approved is False


async def test_edit_applies_field_updates():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-edit"}}
    await graph.ainvoke(_state("hitl-edit"), cfg)
    final = await graph.ainvoke(
        Command(resume={"action": "edit", "edits": {"recommendation": "hold"}}), cfg
    )
    assert final["status"] == "done"
    assert final["assessment"].recommendation == "hold"
    assert final["assessment"].human_approved is True


async def test_bounded_retry_loop_terminates_at_human_gate():
    graph = _wire(RETRY)  # verifier always says "retry" -> loop must be capped
    cfg = {"configurable": {"thread_id": "hitl-loop"}}
    res = await graph.ainvoke(_state("hitl-loop"), cfg)
    assert "__interrupt__" in res  # reached the gate/interrupt, did not run forever
    snap = await graph.aget_state(cfg)
    assert snap.values["status"] == "awaiting_human"
    assert snap.values["retry_count"] >= 2  # loop ran to the cap
