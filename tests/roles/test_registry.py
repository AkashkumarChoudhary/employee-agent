import pytest

from employee_agent.roles.registry import ROLES, get_role
from employee_agent.schemas import RoleConfig


def test_hr_analyst_present():
    rc = get_role("hr_analyst")
    assert isinstance(rc, RoleConfig)
    assert rc.extraction_schema == "CandidateAssessment"
    assert rc.knowledge_namespace == "hr"
    assert rc.system_prompt.strip() != ""


def test_unknown_role_raises():
    with pytest.raises(KeyError):
        get_role("nope")


def test_registry_is_dict():
    assert "hr_analyst" in ROLES
