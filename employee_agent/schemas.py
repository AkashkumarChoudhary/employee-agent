from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    source: str  # e.g. "resume" | "job_description"
    score: float


class SkillMatch(BaseModel):
    requirement: str
    candidate_evidence: str | None = None
    met: bool
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateAssessment(BaseModel):
    candidate_name: str
    years_experience: float
    top_skills: list[str]
    skill_matches: list[SkillMatch]
    overall_match_score: int = Field(ge=0, le=100)
    recommendation: Literal["advance", "hold", "reject"]
    rationale: str
    human_approved: bool = False


class VerifierVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str]
    action: Literal["accept", "retry_retrieval", "retry_analysis"]


class RoleConfig(BaseModel):
    name: str
    system_prompt: str
    extraction_schema: str  # name of the Pydantic model to enforce
    tool_allowlist: list[str] = []
    knowledge_namespace: str


class CreateJobRequest(BaseModel):
    role: str = "hr_analyst"
    job_description: str


class AgentState(TypedDict):
    job_id: str
    role_config: RoleConfig
    job_description: str
    parsed_resume: str
    retrieved_chunks: list[Chunk]
    assessment: CandidateAssessment | None
    verifier_verdict: VerifierVerdict | None
    retry_count: int
    status: Literal["running", "awaiting_human", "done", "error"]
