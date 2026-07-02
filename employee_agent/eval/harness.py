from langgraph.checkpoint.memory import MemorySaver

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role


async def evaluate(provider, cases, role: str = "hr_analyst") -> dict:
    results = []
    for i, case in enumerate(cases):
        retriever = Retriever(provider, VectorStore())
        graph = build_graph(provider, retriever, checkpointer=MemorySaver())
        thread = f"eval-{i}"
        state = new_state(thread, get_role(role), case.job_description, case.resume)
        out = await graph.ainvoke(state, {"configurable": {"thread_id": thread}})
        assessment = out.get("assessment")
        got = assessment.recommendation if assessment else None
        results.append({
            "name": case.name,
            "expected": case.expected_recommendation,
            "got": got,
            "correct": got == case.expected_recommendation,
        })
    correct = sum(r["correct"] for r in results)
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": (correct / len(cases)) if cases else 0.0,
        "results": results,
    }
