import pytest
from pydantic import ValidationError

from employee_agent.schemas import (
    CandidateAssessment,
    Chunk,
    CreateJobRequest,
    RoleConfig,
    SkillMatch,
    VerifierVerdict,
)


def test_candidate_assessment_valid():
    a = CandidateAssessment(
        candidate_name="Ada Lovelace",
        years_experience=5.0,
        top_skills=["python", "math"],
        skill_matches=[
            SkillMatch(requirement="python", candidate_evidence="3 yrs Python",
                       met=True, confidence=0.9)
        ],
        overall_match_score=82,
        recommendation="advance",
        rationale="Strong Python background.",
    )
    assert a.human_approved is False
    assert a.overall_match_score == 82


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        CandidateAssessment(
            candidate_name="X", years_experience=1, top_skills=[], skill_matches=[],
            overall_match_score=150, recommendation="hold", rationale="r",
        )


def test_recommendation_literal_enforced():
    with pytest.raises(ValidationError):
        CandidateAssessment(
            candidate_name="X", years_experience=1, top_skills=[], skill_matches=[],
            overall_match_score=50, recommendation="maybe", rationale="r",
        )


def test_verifier_verdict():
    v = VerifierVerdict(grounded=False, unsupported_claims=["10 yrs exp"],
                        action="retry_analysis")
    assert v.action == "retry_analysis"


def test_create_job_request_defaults():
    r = CreateJobRequest(job_description="Senior Python role")
    assert r.role == "hr_analyst"


def test_role_config_roundtrip():
    rc = RoleConfig(name="hr_analyst", system_prompt="You are...",
                    extraction_schema="CandidateAssessment",
                    tool_allowlist=["verify_certification"],
                    knowledge_namespace="hr")
    assert rc.knowledge_namespace == "hr"


def test_chunk():
    c = Chunk(text="hi", source="resume", score=0.5)
    assert c.source == "resume"
