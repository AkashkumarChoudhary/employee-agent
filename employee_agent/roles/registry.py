from employee_agent.schemas import RoleConfig

HR_ANALYST = RoleConfig(
    name="hr_analyst",
    system_prompt=(
        "You are an experienced HR recruiting analyst. Given a candidate's resume "
        "and a job description, assess fit objectively. Cite evidence from the resume "
        "for each requirement. Never invent experience that is not present. Output "
        "strictly conforms to the CandidateAssessment schema."
    ),
    extraction_schema="CandidateAssessment",
    tool_allowlist=["verify_certification"],
    knowledge_namespace="hr",
)

# Future roles (research/knowledge/support) register here as stubs in later plans.
ROLES: dict[str, RoleConfig] = {
    HR_ANALYST.name: HR_ANALYST,
}


def get_role(name: str) -> RoleConfig:
    return ROLES[name]
