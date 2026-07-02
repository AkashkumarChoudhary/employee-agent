# Employee Agent — Plan 7: Streamlit Demo UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit demo UI over the Plan 5 API — upload a résumé + job description, run the agent, see the draft assessment, approve/edit/reject at the human gate, and view the execution trace — so the whole system is demo-able in a browser in ~3 minutes.

**Architecture:** A new `employee_agent/ui/` package. The HTTP logic lives in a **testable, sync** `EmployeeAgentClient` (httpx) with an injectable transport, so its request-building and response-parsing are unit-tested with `httpx.MockTransport` (no server). A pure `assessment_markdown()` formatter is unit-tested directly. `streamlit_app.py` is a thin view over those two — it reads the API URL/key from env, renders the upload form, and drives create → show draft → approve/edit/reject → trace. A `streamlit.testing.v1.AppTest` smoke test renders the initial screen with no network.

**Tech Stack:** Python 3.11+, streamlit, httpx, plus the Plan 5 API; pytest.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network** — `EmployeeAgentClient` is tested with `httpx.MockTransport`; the Streamlit smoke test renders the initial form only (no API call before submit).
- **Confirmed:** `httpx.MockTransport(handler)` drives a sync `httpx.Client`; `streamlit.testing.v1.AppTest.from_file(path).run()` executes the app with a runtime context.
- The UI depends only on the **public API contract** (Plan 5 endpoints): `POST /jobs` (multipart `job_description`, `role`, `resume`), `GET /jobs/{id}`, `POST /jobs/{id}/approve` (`{action, edits}`), `GET /jobs/{id}/trace`. It never imports the engine directly.
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`. Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: API client and assessment formatter

**Files:**
- Modify: `pyproject.toml` (add `streamlit>=1.30`)
- Create: `employee_agent/ui/__init__.py`
- Create: `employee_agent/ui/client.py`
- Create: `employee_agent/ui/format.py`
- Test: `tests/ui/test_client.py`, `tests/ui/test_format.py`

**Interfaces:**
- `ui.client.EmployeeAgentClient(base_url="http://localhost:8000", api_key="demo-key", client: httpx.Client | None = None)`:
  - `create_job(*, job_description, role, filename, content, content_type="text/plain") -> dict`
  - `get_job(job_id) -> dict`
  - `approve(job_id, action, edits=None) -> dict`
  - `trace(job_id) -> list[dict]`
  - Every method sends the `x-api-key` header and raises `httpx.HTTPStatusError` on non-2xx.
- `ui.format.assessment_markdown(assessment: dict | None) -> str` — a Markdown summary; `"_No assessment yet._"` when `None`.

- [ ] **Step 1: Add dependency to `pyproject.toml`**

Add to the `dependencies` array: `"streamlit>=1.30",`

- [ ] **Step 2: Install**

Run: `.venv/bin/python -m pip install "streamlit>=1.30"`
Expected: installed.

- [ ] **Step 3: Create `employee_agent/ui/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write the failing tests**

`tests/ui/test_client.py`:

```python
import json

import httpx
import pytest

from employee_agent.ui.client import EmployeeAgentClient


def _client(handler, api_key="k"):
    return EmployeeAgentClient(
        api_key=api_key,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test"),
    )


def test_create_job_posts_multipart_with_key():
    seen = {}

    def handler(req):
        seen["path"] = req.url.path
        seen["key"] = req.headers.get("x-api-key")
        seen["ct"] = req.headers.get("content-type", "")
        return httpx.Response(200, json={"job_id": "j1", "status": "awaiting_human"})

    out = _client(handler).create_job(
        job_description="jd", role="hr_analyst", filename="r.txt", content=b"resume"
    )
    assert out == {"job_id": "j1", "status": "awaiting_human"}
    assert seen["path"] == "/jobs"
    assert seen["key"] == "k"
    assert "multipart/form-data" in seen["ct"]


def test_approve_sends_action_json():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"job_id": "j1", "status": "done",
                                         "assessment": {"human_approved": True}})

    out = _client(handler).approve("j1", "approve")
    assert out["status"] == "done"
    assert captured["body"] == {"action": "approve", "edits": {}}


def test_get_job_and_trace():
    def handler(req):
        if req.url.path.endswith("/trace"):
            return httpx.Response(200, json={"job_id": "j1", "steps": [{"step": 0, "node": "manager"}]})
        return httpx.Response(200, json={"job_id": "j1", "status": "awaiting_human", "assessment": None})

    c = _client(handler)
    assert c.get_job("j1")["status"] == "awaiting_human"
    assert c.trace("j1")[0]["node"] == "manager"


def test_raises_on_http_error():
    def handler(req):
        return httpx.Response(401, json={"detail": "nope"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).get_job("j1")
```

`tests/ui/test_format.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/ui -v`
Expected: FAIL with `ModuleNotFoundError` for `employee_agent.ui.client` / `.format`.

- [ ] **Step 6: Implement `employee_agent/ui/client.py`**

```python
import httpx


class EmployeeAgentClient:
    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: str = "demo-key", client: httpx.Client | None = None):
        self._client = client or httpx.Client(base_url=base_url, timeout=60.0)
        self._headers = {"x-api-key": api_key}

    def create_job(self, *, job_description: str, role: str, filename: str,
                   content: bytes, content_type: str = "text/plain") -> dict:
        r = self._client.post(
            "/jobs", headers=self._headers,
            data={"job_description": job_description, "role": role},
            files={"resume": (filename, content, content_type)},
        )
        r.raise_for_status()
        return r.json()

    def get_job(self, job_id: str) -> dict:
        r = self._client.get(f"/jobs/{job_id}", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def approve(self, job_id: str, action: str, edits: dict | None = None) -> dict:
        r = self._client.post(
            f"/jobs/{job_id}/approve", headers=self._headers,
            json={"action": action, "edits": edits or {}},
        )
        r.raise_for_status()
        return r.json()

    def trace(self, job_id: str) -> list[dict]:
        r = self._client.get(f"/jobs/{job_id}/trace", headers=self._headers)
        r.raise_for_status()
        return r.json()["steps"]
```

- [ ] **Step 7: Implement `employee_agent/ui/format.py`**

```python
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
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ui -v`
Expected: PASS (4 + 2 = 6 passed).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml employee_agent/ui/__init__.py employee_agent/ui/client.py employee_agent/ui/format.py tests/ui/
git commit -m "feat: streamlit ui api client and assessment formatter"
```

---

### Task 2: Streamlit demo app

**Files:**
- Create: `employee_agent/ui/streamlit_app.py`
- Test: `tests/ui/test_streamlit_app.py`

**Interfaces:**
- `ui.streamlit_app.get_client() -> EmployeeAgentClient` — reads `EMPLOYEE_AGENT_API_URL` / `EMPLOYEE_AGENT_API_KEY` from env (with defaults).
- `ui.streamlit_app.main()` — renders the demo; called at module load so `streamlit run employee_agent/ui/streamlit_app.py` works.

- [ ] **Step 1: Write the failing test** in `tests/ui/test_streamlit_app.py`

```python
from streamlit.testing.v1 import AppTest


def test_initial_screen_renders_without_network():
    at = AppTest.from_file("employee_agent/ui/streamlit_app.py").run()
    assert not at.exception
    assert any("Employee Agent" in t.value for t in at.title)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/ui/test_streamlit_app.py -v`
Expected: FAIL (file does not exist).

- [ ] **Step 3: Implement `employee_agent/ui/streamlit_app.py`**

```python
import os

import streamlit as st

from employee_agent.ui.client import EmployeeAgentClient
from employee_agent.ui.format import assessment_markdown


def get_client() -> EmployeeAgentClient:
    return EmployeeAgentClient(
        base_url=os.environ.get("EMPLOYEE_AGENT_API_URL", "http://localhost:8000"),
        api_key=os.environ.get("EMPLOYEE_AGENT_API_KEY", "demo-key"),
    )


def main() -> None:
    st.set_page_config(page_title="Employee Agent — HR Analyst", page_icon="🧑‍💼")
    st.title("🧑‍💼 Employee Agent — HR Analyst")
    st.caption(
        "Upload a résumé and a job description; the agent retrieves evidence, "
        "analyzes fit, self-verifies grounding, and pauses for your approval."
    )

    client = get_client()
    ss = st.session_state
    ss.setdefault("job_id", None)
    ss.setdefault("job", None)

    with st.form("new_job"):
        jd = st.text_area("Job description", height=160,
                          placeholder="Senior Python engineer with Django experience...")
        role = st.selectbox("Role preset", ["hr_analyst"])
        resume = st.file_uploader("Résumé (.pdf / .txt / .md)", type=["pdf", "txt", "md"])
        submitted = st.form_submit_button("Run assessment")

    if submitted:
        if not jd or resume is None:
            st.error("Provide both a job description and a résumé file.")
        else:
            with st.spinner("Running the agent…"):
                created = client.create_job(
                    job_description=jd, role=role,
                    filename=resume.name, content=resume.getvalue(),
                )
                ss.job_id = created["job_id"]
                ss.job = client.get_job(ss.job_id)

    if ss.job_id and ss.job:
        st.divider()
        st.subheader(f"Job `{ss.job_id[:8]}` — status: `{ss.job['status']}`")
        st.markdown(assessment_markdown(ss.job.get("assessment")))

        if ss.job["status"] == "awaiting_human":
            st.info("The agent is awaiting your decision.")
            c1, c2, c3 = st.columns(3)
            if c1.button("✅ Approve"):
                ss.job = client.approve(ss.job_id, "approve")
                st.rerun()
            if c2.button("🚫 Reject"):
                ss.job = client.approve(ss.job_id, "reject")
                st.rerun()
            new_rec = c3.selectbox("Edit recommendation", ["advance", "hold", "reject"])
            if c3.button("✏️ Save edit"):
                ss.job = client.approve(ss.job_id, "edit", {"recommendation": new_rec})
                st.rerun()

        with st.expander("Execution trace"):
            for step in client.trace(ss.job_id):
                st.write(f"{step['step']}. `{step['node']}`")


main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/ui/test_streamlit_app.py -v`
Expected: PASS (1 passed).

> **Watch-point:** if `AppTest` reports an exception because `st.rerun`/`st.columns` behave differently, note the initial render has no button pressed, so only the form path runs; the smoke test should stay green. If `at.title` is empty, assert on `at.markdown`/`at.header` text instead.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + Plan 7; ~90 passed).

- [ ] **Step 6: Commit**

```bash
git add employee_agent/ui/streamlit_app.py tests/ui/test_streamlit_app.py
git commit -m "feat: streamlit demo app (upload -> assess -> HITL approve -> trace)"
```

---

## Definition of Done (Plan 7)

- `.venv/bin/python -m pytest -q` is green across the UI client, formatter, and app smoke test, plus all prior modules.
- `EmployeeAgentClient` correctly calls every Plan 5 endpoint (verified via `MockTransport`) and raises on HTTP errors.
- `streamlit run employee_agent/ui/streamlit_app.py` renders the upload form; the initial screen renders under `AppTest` with no network.
- The UI talks only to the public API (no engine imports).

## How to run the demo

```bash
# terminal 1 — API
.venv/bin/uvicorn employee_agent.api.app:create_app --factory --reload
# terminal 2 — UI (point it at the API)
EMPLOYEE_AGENT_API_URL=http://localhost:8000 EMPLOYEE_AGENT_API_KEY=demo-key \
  .venv/bin/streamlit run employee_agent/ui/streamlit_app.py
```

## Next Plan

Plan 8 — **Package & document**: Dockerfile + `docker compose up` (app API + optional UI), a README with the architecture diagram and the tech-stack tier table, a Cloud Run service config + deploy script, and GKE/Apigee/Cloud Armor scale-path notes. Finishes the capstone as reproducible and documented.
