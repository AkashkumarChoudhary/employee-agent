from employee_agent.eval.dataset import EvalCase
from employee_agent.eval.harness import evaluate
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict


class KeywordProvider(FakeProvider):
    """Deterministic: rejects a résumé mentioning 'graphic design', else advances."""

    async def generate_structured(self, *, system, prompt, schema):
        if schema is VerifierVerdict:
            return VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
        rec = "reject" if "graphic design" in prompt.lower() else "advance"
        return CandidateAssessment(
            candidate_name="Candidate", years_experience=3.0,
            top_skills=["python"] if rec == "advance" else [],
            skill_matches=[], overall_match_score=75 if rec == "advance" else 20,
            recommendation=rec, rationale="kw",
        )


PY = EvalCase("py", "I have 5 years of Python and Django.", "Senior Python engineer", "advance")
DESIGN = EvalCase("design", "I do graphic design in Photoshop.", "Senior Python engineer", "reject")


async def test_eval_scores_correct_recommendations():
    report = await evaluate(KeywordProvider(), [PY, DESIGN])
    assert report["total"] == 2
    assert report["correct"] == 2
    assert report["accuracy"] == 1.0


async def test_eval_reports_mismatch():
    mislabeled = EvalCase("bad", "I do graphic design.", "Python role", "advance")
    report = await evaluate(KeywordProvider(), [mislabeled])
    assert report["accuracy"] == 0.0
    assert report["results"][0]["got"] == "reject"
