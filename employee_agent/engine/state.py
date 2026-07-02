from employee_agent.schemas import AgentState, RoleConfig


def new_state(
    job_id: str, role_config: RoleConfig, job_description: str, resume_text: str
) -> AgentState:
    return {
        "job_id": job_id,
        "role_config": role_config,
        "job_description": job_description,
        "parsed_resume": resume_text,
        "retrieved_chunks": [],
        "assessment": None,
        "verifier_verdict": None,
        "retry_count": 0,
        "status": "running",
    }
