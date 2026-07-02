from dataclasses import dataclass


@dataclass
class EvalCase:
    name: str
    resume: str
    job_description: str
    expected_recommendation: str  # "advance" | "hold" | "reject"


# A tiny labeled set to sanity-check a real LLM ("metrics, not vibe-checks").
LABELED_CASES: list[EvalCase] = [
    EvalCase(
        "strong-python",
        "Senior engineer with 8 years of Python, Django, and PostgreSQL. Led API teams.",
        "Senior Python engineer with Django and REST APIs.",
        "advance",
    ),
    EvalCase(
        "career-switch",
        "Graphic designer for 6 years; recently completed a 3-month Python bootcamp.",
        "Senior Python engineer with 5+ years backend experience.",
        "hold",
    ),
    EvalCase(
        "wrong-field",
        "Registered nurse with 10 years in critical care. No software experience.",
        "Senior Python engineer.",
        "reject",
    ),
    EvalCase(
        "data-scientist",
        "Data scientist, 5 years Python, pandas, scikit-learn, some FastAPI services.",
        "Backend Python engineer building FastAPI services.",
        "advance",
    ),
    EvalCase(
        "junior",
        "New grad, 1 internship writing Python scripts. Eager to learn.",
        "Senior Python engineer with 5+ years and team leadership.",
        "reject",
    ),
]
