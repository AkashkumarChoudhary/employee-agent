from employee_agent.providers.base import Provider
from employee_agent.rag import ingest
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState, CandidateAssessment

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
        return {"assessment": assessment, "status": "done"}

    return analyst
