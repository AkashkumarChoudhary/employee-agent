def assessment_markdown(assessment: dict | None) -> str:
    if not assessment:
        return "_No assessment yet._"
    skills = ", ".join(assessment.get("top_skills", [])) or "—"
    review = "✅ approved" if assessment.get("human_approved") else "⏳ pending"
    return "\n".join([
        f"### {assessment.get('candidate_name', 'Candidate')}",
        f"- **Recommendation:** `{assessment.get('recommendation')}`",
        f"- **Match score:** {assessment.get('overall_match_score')}/100",
        f"- **Experience:** {assessment.get('years_experience')} yrs",
        f"- **Top skills:** {skills}",
        f"- **Human review:** {review}",
        "",
        assessment.get("rationale", ""),
    ])
