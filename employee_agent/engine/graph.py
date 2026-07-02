from langgraph.graph import END, START, StateGraph

from employee_agent.engine.nodes import (
    make_analyst,
    make_finalizer,
    make_gate,
    make_hitl,
    make_manager,
    make_parser,
    make_retriever_node,
    make_verifier,
)
from employee_agent.engine.routing import route_after_verifier
from employee_agent.providers.base import Provider
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState


def build_graph(provider: Provider, retriever: Retriever, checkpointer=None, tool_client=None):
    g = StateGraph(AgentState)
    g.add_node("manager", make_manager())
    g.add_node("parser", make_parser(retriever))
    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("analyst", make_analyst(provider, tool_client=tool_client))
    g.add_node("verifier", make_verifier(provider))
    g.add_node("gate", make_gate())
    g.add_node("hitl", make_hitl())
    g.add_node("finalizer", make_finalizer())

    g.add_edge(START, "manager")
    g.add_edge("manager", "parser")
    g.add_edge("parser", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"retriever": "retriever", "analyst": "analyst", "gate": "gate"},
    )
    g.add_edge("gate", "hitl")
    g.add_edge("hitl", "finalizer")
    g.add_edge("finalizer", END)
    return g.compile(checkpointer=checkpointer)
