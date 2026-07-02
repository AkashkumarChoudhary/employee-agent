from employee_agent.ui.format import assessment_markdown


def test_none_assessment():
    assert "No assessment" in assessment_markdown(None)


def test_renders_key_fields():
    md = assessment_markdown({
        "candidate_name": "Ada Lovelace", "recommendation": "advance",
        "overall_match_score": 88, "years_experience": 5.0,
        "top_skills": ["python", "django"], "human_approved": True,
        "rationale": "Strong Python background.",
    })
    assert "Ada Lovelace" in md
    assert "advance" in md
    assert "88/100" in md
    assert "python" in md
    assert "Strong Python background." in md
