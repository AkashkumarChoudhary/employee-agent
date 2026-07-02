from langgraph.graph import END, START, StateGraph

from employee_agent.engine.nodes import (
    make_analyst,
    make_manager,
    make_parser,
    make_retriever_node,
)
from employee_agent.providers.base import Provider
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState


def build_graph(provider: Provider, retriever: Retriever, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("manager", make_manager())
    g.add_node("parser", make_parser(retriever))
    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("analyst", make_analyst(provider))
    g.add_edge(START, "manager")
    g.add_edge("manager", "parser")
    g.add_edge("parser", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", END)
    return g.compile(checkpointer=checkpointer)
