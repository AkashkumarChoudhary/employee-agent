from employee_agent.schemas import AgentState

MAX_RETRIES = 2


def route_after_verifier(state: AgentState) -> str:
    verdict = state["verifier_verdict"]
    if verdict.action == "accept" or state["retry_count"] > MAX_RETRIES:
        return "gate"
    if verdict.action == "retry_retrieval":
        return "retriever"
    return "analyst"
