# Employee Agent — Plan 8: Package & Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the capstone as reproducible and documented — a light **eval harness** (metrics, not vibe-checks), a **Dockerfile** + `docker compose up` for the API and UI, a **README** with the architecture and tech-stack tier table, a **Cloud Run** deploy config + **GKE/Apigee/Cloud Armor** scale-path notes, and a **GitHub Actions** CI workflow.

**Architecture:** One new tested module, `employee_agent/eval/`: a small labeled dataset plus an `evaluate()` harness that runs each case through the real graph and tallies recommendation accuracy — unit-tested deterministically with a keyword provider. Everything else is packaging/config/docs (the spec's 🔵 tier: deploy-ready, not operated): `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `README.md`, `deploy/` (Cloud Run + scale-path notes), and `.github/workflows/ci.yml`.

**Tech Stack:** Python 3.11+, Docker/Compose, GitHub Actions; reuses `build_graph`, `new_state`, `Retriever`, `get_role`.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network** — the eval harness test uses an in-process keyword provider + ephemeral store; no LLM calls.
- **Honesty rule (from the spec):** 🔵 items (Cloud Run, GKE/Apigee/Cloud Armor, CI) are config + docs, labeled "deploy-ready / designed-for", never "running in production."
- The uvicorn entrypoint is the factory `employee_agent.api.app:create_app` (`--factory`); the UI entrypoint is `employee_agent/ui/streamlit_app.py`.
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`. Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: Light eval harness

**Files:**
- Create: `employee_agent/eval/__init__.py`
- Create: `employee_agent/eval/dataset.py`
- Create: `employee_agent/eval/harness.py`
- Test: `tests/eval/test_harness.py`

**Interfaces:**
- `eval.dataset.EvalCase(name: str, resume: str, job_description: str, expected_recommendation: str)` and `eval.dataset.LABELED_CASES: list[EvalCase]` (~5 realistic cases for real-LLM runs).
- `eval.harness.evaluate(provider, cases, role: str = "hr_analyst") -> dict` — runs each case through a freshly built graph (own ephemeral store + `MemorySaver`, namespaced `eval-<i>`) to the human gate, reads the draft `assessment.recommendation`, and returns `{"total", "correct", "accuracy", "results": [{"name","expected","got","correct"}, ...]}`.

- [ ] **Step 1: Write the failing test** in `tests/eval/test_harness.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/eval/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.eval.harness'`

- [ ] **Step 3: Create `employee_agent/eval/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Implement `employee_agent/eval/dataset.py`**

```python
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
```

- [ ] **Step 5: Implement `employee_agent/eval/harness.py`**

```python
from langgraph.checkpoint.memory import MemorySaver

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role


async def evaluate(provider, cases, role: str = "hr_analyst") -> dict:
    results = []
    for i, case in enumerate(cases):
        retriever = Retriever(provider, VectorStore())
        graph = build_graph(provider, retriever, checkpointer=MemorySaver())
        thread = f"eval-{i}"
        state = new_state(thread, get_role(role), case.job_description, case.resume)
        out = await graph.ainvoke(state, {"configurable": {"thread_id": thread}})
        assessment = out.get("assessment")
        got = assessment.recommendation if assessment else None
        results.append({
            "name": case.name,
            "expected": case.expected_recommendation,
            "got": got,
            "correct": got == case.expected_recommendation,
        })
    correct = sum(r["correct"] for r in results)
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": (correct / len(cases)) if cases else 0.0,
        "results": results,
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/eval/test_harness.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + eval; ~92 passed)

- [ ] **Step 8: Commit**

```bash
git add employee_agent/eval/ tests/eval/
git commit -m "feat: light eval harness with labeled candidate dataset"
```

---

### Task 2: Docker, README, deploy config, and CI

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `docker-compose.yml`
- Create: `README.md`
- Create: `deploy/cloud-run.md`
- Create: `.github/workflows/ci.yml`

This task is packaging/config/docs (the spec's 🔵 tier). There is no unit test; verification is structural (files present, compose config parses if Docker is available, README covers the required sections).

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; wheels cover the rest.
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install \
    "pydantic>=2.6" "pydantic-settings>=2.2" "google-genai>=0.3" "ollama>=0.3" \
    "langchain-text-splitters>=0.2" "pypdf>=4.0" "chromadb>=0.5" \
    "langgraph>=0.2" "langgraph-checkpoint-sqlite>=2.0" \
    "fastapi>=0.110" "python-multipart>=0.0.9" "slowapi>=0.1.9" "uvicorn>=0.29" \
    "mcp>=1.0" "streamlit>=1.30"

COPY employee_agent ./employee_agent

EXPOSE 8000
CMD ["uvicorn", "employee_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create `.dockerignore`**

```gitignore
.venv/
data/
__pycache__/
*.pyc
.pytest_cache/
.git/
tests/
docs/
*.md
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      PROVIDER: ${PROVIDER:-gemini}
      GEMINI_API_KEY: ${GEMINI_API_KEY:-}
      ENABLE_FAILOVER: "false"
      API_KEYS: ${API_KEYS:-demo-key}
      SQLITE_PATH: /data/employee_agent.db
      CHROMA_PATH: /data/chroma
    volumes:
      - agent-data:/data

  ui:
    build: .
    command: streamlit run employee_agent/ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ports:
      - "8501:8501"
    environment:
      EMPLOYEE_AGENT_API_URL: http://api:8000
      EMPLOYEE_AGENT_API_KEY: ${API_KEYS:-demo-key}
    depends_on:
      - api

volumes:
  agent-data:
```

> Chroma runs **embedded** (a local persistent path on the shared volume), not as a separate service — so `docker compose up` brings up just API + UI.

- [ ] **Step 4: Create `README.md`**

Write a README covering, in this order:
1. **Title + one-liner:** "Employee Agent — a configurable digital-employee engine (HR analyst hero use case)."
2. **What it is:** a LangGraph orchestrator-worker graph (RAG → analysis → CRAG/Self-RAG verification → bounded retry → human-in-the-loop → finalize), served over FastAPI, demoed in Streamlit, with a real MCP tool. Swap a `RoleConfig` to retarget the same engine.
3. **Architecture diagram** (fenced ` ```text ` block):

```text
[Streamlit UI] --HTTP--> [FastAPI: auth, BOLA, rate-limit]
                               |
                               v
   manager -> parser -> retriever(RAG/Chroma) -> analyst --(MCP verify_certification)
                                                     |
                                                     v
                          verifier(CRAG/Self-RAG) --accept--> gate -> hitl(interrupt) -> finalizer
                                |  ^ bounded retry (<= MAX_RETRIES)
                                v  |
                          retriever / analyst
   (state persisted + resumable via SQLite checkpointer)
```

4. **Tech-stack tier table** — reproduce the spec's §6 table (🟢 fully built · 🟡 real but minimal · 🔵 config/docs) so it is honest about what runs vs. what is designed.
5. **Quickstart:**

````markdown
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install pydantic pydantic-settings google-genai ollama langchain-text-splitters \
  pypdf chromadb langgraph langgraph-checkpoint-sqlite fastapi python-multipart \
  slowapi uvicorn mcp streamlit pytest pytest-asyncio httpx
pytest -q                      # 90+ tests, fully offline

# Run the API + UI locally
export PROVIDER=fake           # or PROVIDER=gemini GEMINI_API_KEY=...
uvicorn employee_agent.api.app:create_app --factory --reload
EMPLOYEE_AGENT_API_URL=http://localhost:8000 EMPLOYEE_AGENT_API_KEY=demo-key \
  streamlit run employee_agent/ui/streamlit_app.py

# Or with Docker
docker compose up --build
```
````

6. **API reference:** the six endpoints table (from the spec §8) + the `X-API-Key` note.
7. **Security:** X-API-Key auth, per-key BOLA ownership, slowapi rate limiting, upload validation, MCP allowlist (SSRF mitigation).
8. **Testing:** `pytest -q` is fully offline/deterministic (`FakeProvider`, `temperature=0`, no model downloads); light eval harness in `employee_agent/eval/`.
9. **Deploy & scale path:** point to `deploy/cloud-run.md`; state clearly that Cloud Run/GKE/Apigee/Cloud Armor are **designed-for, not running**.

- [ ] **Step 5: Create `deploy/cloud-run.md`**

Document the production target and scale path (🔵 — not operated):
- **Cloud Run:** `gcloud run deploy employee-agent --source . --region <r> --set-env-vars PROVIDER=gemini,...` with a Secret Manager reference for `GEMINI_API_KEY`; swap the provider abstraction to **Vertex AI** with no app changes.
- **Scale path:** GKE + multi-cluster Inference Gateway for global routing; **Apigee** for quota + Spike-Arrest; **Cloud Armor** WAF for L3/L4 DDoS. Note the in-graph hard iteration cap (`MAX_RETRIES`) as the app-level runaway-loop guard.

- [ ] **Step 6: Create `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
    branches: [ "**" ]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install pydantic pydantic-settings google-genai ollama \
            langchain-text-splitters pypdf chromadb langgraph \
            langgraph-checkpoint-sqlite fastapi python-multipart slowapi \
            uvicorn mcp streamlit pytest pytest-asyncio httpx
      - name: Test
        run: pytest -q
```

- [ ] **Step 7: Verify structurally**

Run:
```bash
ls Dockerfile .dockerignore docker-compose.yml README.md deploy/cloud-run.md .github/workflows/ci.yml
docker compose config >/dev/null 2>&1 && echo "compose OK" || echo "compose check skipped (docker not available)"
grep -qi "tier" README.md && grep -qi "quickstart\|Quickstart\|## Quick" README.md && echo "README sections present"
```
Expected: all files listed; README section check prints "README sections present".

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore docker-compose.yml README.md deploy/ .github/
git commit -m "docs: Dockerfile, compose, README, Cloud Run + scale-path notes, CI"
```

---

## Definition of Done (Plan 8)

- `.venv/bin/python -m pytest -q` is green including the eval harness (~92 tests).
- `evaluate(provider, cases)` runs cases through the real graph and reports accuracy; a mismatch lowers accuracy (verified deterministically).
- `Dockerfile` + `docker-compose.yml` bring up API + UI (`docker compose up`); Chroma is embedded.
- `README.md` documents the architecture, the tech-stack tier table (honest about 🟢/🟡/🔵), quickstart, API, security, testing, and the deploy/scale path.
- `deploy/cloud-run.md` and `.github/workflows/ci.yml` exist and are labeled deploy-ready (not operated).

## Project Complete

With Plan 8 done, the capstone is reproducible (`docker compose up`), demo-able (Streamlit over FastAPI), tested (90+ offline tests), and honest (tiered tech-stack table). The remaining role presets (research/knowledge/support) and a live cloud deploy are documented as future work.
