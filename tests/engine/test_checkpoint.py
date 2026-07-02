from employee_agent.engine.checkpoint import sqlite_checkpointer
from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=90, recommendation="advance",
    rationale="ok",
)


async def test_sqlite_checkpoint_persists_and_reloads(tmp_path):
    db = str(tmp_path / "ckpt.sqlite")
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    retriever = Retriever(provider, VectorStore())
    cfg = {"configurable": {"thread_id": "j-sql"}}
    state = new_state("j-sql", get_role("hr_analyst"),
                      "Senior Python engineer.", "Ada Lovelace. Python, Django.")

    async with sqlite_checkpointer(db) as saver:
        graph = build_graph(provider, retriever, checkpointer=saver)
        await graph.ainvoke(state, cfg)
        snap = await graph.aget_state(cfg)
        assert snap.values["status"] == "done"

    # Re-open a fresh saver on the same file: state persisted to disk.
    async with sqlite_checkpointer(db) as saver2:
        graph2 = build_graph(provider, retriever, checkpointer=saver2)
        snap2 = await graph2.aget_state(cfg)
        assert snap2.values["assessment"] == CANNED
