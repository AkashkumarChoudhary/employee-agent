from langgraph.types import interrupt

from employee_agent.providers.base import Provider
from employee_agent.rag import ingest
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState, CandidateAssessment, VerifierVerdict

RESUME_CHUNK_SIZE = 800
RESUME_CHUNK_OVERLAP = 100
RETRIEVE_K = 6


def make_manager():
    async def manager(state: AgentState) -> dict:
        return {"status": "running", "retry_count": state.get("retry_count", 0)}

    return manager


def make_parser(retriever: Retriever):
    async def parser(state: AgentState) -> dict:
        text = " ".join(state["parsed_resume"].split())
        chunks = ingest.split_text(
            text, source="resume",
            chunk_size=RESUME_CHUNK_SIZE, chunk_overlap=RESUME_CHUNK_OVERLAP,
        )
        await retriever.index(state["job_id"], chunks)
        return {"parsed_resume": text}

    return parser


def make_retriever_node(retriever: Retriever):
    async def retriever_node(state: AgentState) -> dict:
        chunks = await retriever.retrieve(
            state["job_id"], query=state["job_description"], k=RETRIEVE_K
        )
        return {"retrieved_chunks": chunks}

    return retriever_node


def make_analyst(provider: Provider):
    async def analyst(state: AgentState) -> dict:
        role = state["role_config"]
        evidence = "\n".join(f"- {c.text}" for c in state["retrieved_chunks"])
        prompt = (
            f"Job description:\n{state['job_description']}\n\n"
            f"Relevant resume evidence:\n{evidence}\n\n"
            "Assess the candidate against the job description. Cite evidence; "
            "do not invent experience."
        )
        assessment = await provider.generate_structured(
            system=role.system_prompt, prompt=prompt, schema=CandidateAssessment
        )
        return {"assessment": assessment}

    return analyst


def make_verifier(provider: Provider):
    async def verifier(state: AgentState) -> dict:
        a = state["assessment"]
        evidence = "\n".join(f"- {c.text}" for c in state["retrieved_chunks"])
        prompt = (
            "Candidate assessment to check:\n"
            f"- recommendation: {a.recommendation}\n"
            f"- rationale: {a.rationale}\n"
            f"- top_skills: {a.top_skills}\n\n"
            f"Source evidence (retrieved resume chunks):\n{evidence}\n\n"
            "Are the assessment's claims grounded in the evidence? If not, decide "
            "whether to retry retrieval or retry analysis."
        )
        verdict = await provider.generate_structured(
            system=(
                "You are a strict grounding checker implementing CRAG/Self-RAG. "
                "Only accept claims supported by the evidence."
            ),
            prompt=prompt,
            schema=VerifierVerdict,
        )
        update = {"verifier_verdict": verdict}
        if verdict.action != "accept":
            update["retry_count"] = state["retry_count"] + 1
        return update

    return verifier


def make_gate():
    async def gate(state: AgentState) -> dict:
        return {"status": "awaiting_human"}

    return gate


def make_hitl():
    async def hitl(state: AgentState) -> dict:
        decision = interrupt(
            {
                "assessment": state["assessment"].model_dump(),
                "message": "Approve, edit, or reject this candidate assessment.",
            }
        ) or {}
        action = decision.get("action", "approve")
        assessment = state["assessment"]
        if action == "edit":
            assessment = assessment.model_copy(update=decision.get("edits", {}))
            assessment = assessment.model_copy(update={"human_approved": True})
        elif action == "reject":
            assessment = assessment.model_copy(update={"human_approved": False})
        else:  # approve
            assessment = assessment.model_copy(update={"human_approved": True})
        return {"assessment": assessment}

    return hitl


def make_finalizer():
    async def finalizer(state: AgentState) -> dict:
        return {"status": "done" if state["assessment"].human_approved else "error"}

    return finalizer
