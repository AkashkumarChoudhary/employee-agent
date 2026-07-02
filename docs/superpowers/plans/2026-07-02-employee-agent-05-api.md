# Employee Agent — Plan 5: FastAPI Serving Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the compiled graph in a production-shaped FastAPI app — `POST /jobs` (resume upload + JD → run the graph to the human gate → return `job_id`), `GET /jobs/{id}`, `POST /jobs/{id}/approve` (resume via `Command`), `GET /jobs/{id}/trace`, `GET /health`, `GET /whoami` — secured with `X-API-Key` auth, per-key ownership (BOLA) checks, slowapi rate limiting, and Pydantic contracts.

**Architecture:** A new `employee_agent/api/` package with an app factory `create_app(settings, provider)` so tests inject a `FakeProvider` and temp paths. The provider and a persistent `Retriever` live on `app.state`; a `JobStore` (in-memory `job_id → owner/role/status`) enforces ownership. The graph state itself is durably persisted by the existing SQLite checkpointer, so each request opens a short-lived `AsyncSqliteSaver` on the shared `sqlite_path` (proven in Plan 3/4). Endpoints live on a module-level `APIRouter` and read dependencies via `request.app.state`; slowapi's `limiter` is module-level with a per-app dynamic limit.

**Tech Stack:** Python 3.11+, FastAPI, `python-multipart` (uploads), slowapi (rate limiting), uvicorn (serving), httpx (async test client), plus all prior modules.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network** — inject `FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})`; temp `sqlite_path`/`chroma_path` per test.
- **Confirmed facts (probed):** `job_id = uuid.uuid4().hex` (32 chars) is a valid chroma collection name (chroma requires 3–512 chars of `[a-zA-Z0-9._-]`); a graph run pauses at HITL so the result contains `"__interrupt__"` → report `status="awaiting_human"`; `graph.aget_state_history(cfg)` yields snapshots whose `.next[0]` gives the pending node (`__start__, manager, parser, retriever, analyst, verifier, gate, hitl`) — walk `reversed(history)` for the execution path; slowapi `Limiter.limit` accepts a `Callable[..., str]`.
- Reuse verbatim: `build_graph`, `new_state`, `sqlite_checkpointer`, `get_role`, `ingest.load_document`, `build_provider`, `Retriever`, `VectorStore`, `Settings`/`get_settings`, `CandidateAssessment`, `VerifierVerdict`.
- API tests use `httpx.AsyncClient(transport=ASGITransport(app=app))` in async test functions (keeps the app's async DB in one event loop).
- Ownership failures return **404** (never reveal another owner's job exists). Auth failures return **401**.
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`. Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: Config, auth, JobStore, and app skeleton

**Files:**
- Modify: `pyproject.toml` (add `fastapi`, `python-multipart`, `slowapi`, `uvicorn`; `httpx` to dev)
- Modify: `employee_agent/config.py` (add `api_keys`, `rate_limit`, `allowed_api_keys()`)
- Create: `employee_agent/api/__init__.py`
- Create: `employee_agent/api/store.py`
- Create: `employee_agent/api/auth.py`
- Create: `employee_agent/api/app.py` (skeleton: `create_app`, `/health`, `/whoami`)
- Test: `tests/api/test_store.py`, `tests/api/test_app_basics.py`

**Interfaces:**
- `Settings.api_keys: str = "demo-key"` (comma-separated) and `Settings.allowed_api_keys() -> set[str]`; `Settings.rate_limit: str = "100/minute"`.
- `api.store.JobStore` — `create(job_id, owner, role, status="running") -> JobRecord`, `get(job_id) -> JobRecord | None`, `set_status(job_id, status)`. `JobRecord(job_id, owner, role, status)`.
- `api.auth.require_api_key(request, x_api_key: Header) -> str` — FastAPI dependency; 401 unless the header is a known key.
- `api.app.create_app(settings=None, provider=None) -> FastAPI` — wires `app.state.settings/provider/retriever/jobs/limiter`; mounts the router; `GET /health` (open) and `GET /whoami` (authed).

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

Add to the `dependencies` array:

```toml
    "fastapi>=0.110",
    "python-multipart>=0.0.9",
    "slowapi>=0.1.9",
    "uvicorn>=0.29",
```

And extend the dev extras:

```toml
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "httpx>=0.27"]
```

- [ ] **Step 2: Install**

Run: `.venv/bin/python -m pip install "fastapi>=0.110" "python-multipart>=0.0.9" "slowapi>=0.1.9" "uvicorn>=0.29" "httpx>=0.27"`
Expected: installs successfully.

- [ ] **Step 3: Extend `employee_agent/config.py`**

Add these two fields to `Settings` (after `chroma_path`):

```python
    api_keys: str = "demo-key"
    rate_limit: str = "100/minute"
```

And add this method to `Settings`:

```python
    def allowed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}
```

- [ ] **Step 4: Write the failing tests**

`tests/api/test_store.py`:

```python
from employee_agent.api.store import JobStore


def test_create_and_get():
    store = JobStore()
    store.create("j1", owner="key-a", role="hr_analyst")
    rec = store.get("j1")
    assert rec.owner == "key-a"
    assert rec.status == "running"


def test_set_status():
    store = JobStore()
    store.create("j1", owner="key-a", role="hr_analyst")
    store.set_status("j1", "done")
    assert store.get("j1").status == "done"


def test_get_missing_returns_none():
    assert JobStore().get("nope") is None
```

`tests/api/test_app_basics.py`:

```python
import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider


def _app(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a,key-b",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    return create_app(s, provider=FakeProvider())


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_open(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


async def test_whoami_requires_key(tmp_path):
    async with _client(_app(tmp_path)) as c:
        assert (await c.get("/whoami")).status_code == 401
        assert (await c.get("/whoami", headers={"x-api-key": "bad"})).status_code == 401
        ok = await c.get("/whoami", headers={"x-api-key": "key-a"})
        assert ok.status_code == 200
        assert ok.json()["api_key"] == "key-a"
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/api -v`
Expected: FAIL with `ModuleNotFoundError` for `employee_agent.api.store` / `employee_agent.api.app`.

- [ ] **Step 6: Create `employee_agent/api/__init__.py`** (empty)

```python
```

- [ ] **Step 7: Implement `employee_agent/api/store.py`**

```python
from dataclasses import dataclass


@dataclass
class JobRecord:
    job_id: str
    owner: str
    role: str
    status: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create(self, job_id: str, owner: str, role: str, status: str = "running") -> JobRecord:
        rec = JobRecord(job_id=job_id, owner=owner, role=role, status=status)
        self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is not None:
            rec.status = status
```

- [ ] **Step 8: Implement `employee_agent/api/auth.py`**

```python
from fastapi import Header, HTTPException, Request


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> str:
    allowed = request.app.state.settings.allowed_api_keys()
    if not x_api_key or x_api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return x_api_key
```

- [ ] **Step 9: Implement `employee_agent/api/app.py` (skeleton)**

```python
from fastapi import APIRouter, Depends, FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from employee_agent.api.auth import require_api_key
from employee_agent.api.store import JobStore
from employee_agent.config import Settings, get_settings
from employee_agent.providers.factory import build_provider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore

router = APIRouter()


def _rate_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (
        request.client.host if request.client else "anon"
    )


def _limit_value(request: Request) -> str:
    return request.app.state.settings.rate_limit


limiter = Limiter(key_func=_rate_key)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/whoami")
async def whoami(api_key: str = Depends(require_api_key)):
    return {"api_key": api_key}


def create_app(settings: Settings | None = None, provider=None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Employee Agent API")
    prov = provider or build_provider(s)
    app.state.settings = s
    app.state.provider = prov
    app.state.retriever = Retriever(prov, VectorStore(path=s.chroma_path))
    app.state.jobs = JobStore()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    return app
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/api -v`
Expected: PASS (3 + 2 = 5 passed).

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml employee_agent/config.py employee_agent/api/ tests/api/
git commit -m "feat: api skeleton (create_app, X-API-Key auth, JobStore, health/whoami)"
```

---

### Task 2: Create & fetch jobs (with BOLA ownership and upload validation)

**Files:**
- Create: `employee_agent/api/schemas.py`
- Modify: `employee_agent/api/app.py` (add `POST /jobs`, `GET /jobs/{id}`, graph runner helpers)
- Test: `tests/api/test_jobs.py`

**Interfaces:**
- `api.schemas.CreateJobResponse(job_id: str, status: str)`, `JobResponse(job_id: str, status: str, assessment: dict | None = None)`.
- `POST /jobs` (multipart: `job_description` Form, `role` Form default `hr_analyst`, `resume` File; `X-API-Key`) → validates suffix ∈ `{.pdf,.txt,.md}` (415) and size ≤ 5 MB (413) and known role (422); creates `job_id=uuid4().hex`, records ownership, runs the graph to the human gate, returns `CreateJobResponse` (`status="awaiting_human"`).
- `GET /jobs/{id}` (`X-API-Key`) → ownership-checked (404 if missing/not owner); returns `JobResponse` with the current assessment from the checkpoint.

- [ ] **Step 1: Write the failing test** in `tests/api/test_jobs.py`

```python
import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


def _app(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a,key-b",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    prov = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    return create_app(s, provider=prov)


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _files():
    return {"resume": ("resume.txt", b"Ada Lovelace. 5 years Python and Django.", "text/plain")}


async def test_create_job_pauses_for_human(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "Senior Python engineer", "role": "hr_analyst"},
                         files=_files())
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "awaiting_human"
        assert body["job_id"]


async def test_get_job_returns_assessment(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-a"})
        assert r.status_code == 200
        assert r.json()["assessment"]["candidate_name"] == "Ada Lovelace"


async def test_bola_other_key_cannot_read(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-b"})
        assert r.status_code == 404


async def test_rejects_unsupported_file_type(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "x"},
                         files={"resume": ("resume.exe", b"nope", "application/octet-stream")})
        assert r.status_code == 415


async def test_missing_job_description_is_422(tmp_path):
    async with _client(_app(tmp_path)) as c:
        r = await c.post("/jobs", headers={"x-api-key": "key-a"}, files=_files())
        assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/api/test_jobs.py -v`
Expected: FAIL (routes/`schemas` not present → 404s / ImportError).

- [ ] **Step 3: Implement `employee_agent/api/schemas.py`**

```python
from pydantic import BaseModel


class CreateJobResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    assessment: dict | None = None
```

- [ ] **Step 4: Add imports and helpers to `employee_agent/api/app.py`**

Extend the imports at the top:

```python
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import File, Form, HTTPException, UploadFile

from employee_agent.api.schemas import CreateJobResponse, JobResponse
from employee_agent.engine.checkpoint import sqlite_checkpointer
from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.rag import ingest
from employee_agent.roles.registry import get_role

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
```

Add these helpers (above `create_app`):

```python
def _thread(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}


def _status_from_result(result: dict) -> str:
    if "__interrupt__" in result:
        return "awaiting_human"
    return result.get("status", "done")


def _owned_or_404(request: Request, job_id: str, api_key: str):
    rec = request.app.state.jobs.get(job_id)
    if rec is None or rec.owner != api_key:
        raise HTTPException(status_code=404, detail="job not found")
    return rec
```

- [ ] **Step 5: Add the `POST /jobs` and `GET /jobs/{id}` routes to `employee_agent/api/app.py`**

Add (after the `whoami` route, before `create_app`):

```python
@router.post("/jobs", response_model=CreateJobResponse)
@limiter.limit(_limit_value)
async def create_job(
    request: Request,
    job_description: str = Form(...),
    role: str = Form("hr_analyst"),
    resume: UploadFile = File(...),
    api_key: str = Depends(require_api_key),
):
    suffix = Path(resume.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"unsupported file type: {suffix!r}")
    data = await resume.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large")
    try:
        get_role(role)
    except KeyError:
        raise HTTPException(status_code=422, detail=f"unknown role: {role!r}")

    fd, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        resume_text = ingest.load_document(tmp)
    finally:
        os.unlink(tmp)

    job_id = uuid.uuid4().hex
    request.app.state.jobs.create(job_id, owner=api_key, role=role)
    state = new_state(job_id, get_role(role), job_description, resume_text)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = build_graph(request.app.state.provider, request.app.state.retriever, checkpointer=saver)
        result = await graph.ainvoke(state, _thread(job_id))
    status = _status_from_result(result)
    request.app.state.jobs.set_status(job_id, status)
    return CreateJobResponse(job_id=job_id, status=status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    rec = _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = build_graph(request.app.state.provider, request.app.state.retriever, checkpointer=saver)
        snap = await graph.aget_state(_thread(job_id))
    assessment = snap.values.get("assessment")
    return JobResponse(
        job_id=job_id, status=rec.status,
        assessment=assessment.model_dump() if assessment else None,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/api/test_jobs.py -v`
Expected: PASS (5 passed).

> **Watch-point:** if `httpx`/`ASGITransport` raises on lifespan, pass `ASGITransport(app=app)` as shown (no lifespan needed). If multiple `PersistentClient`s at different `chroma_path`s conflict across tests, they should not (keyed by path); if they do, switch tests to unique subdirs (already done via `tmp_path`).

- [ ] **Step 7: Commit**

```bash
git add employee_agent/api/schemas.py employee_agent/api/app.py tests/api/test_jobs.py
git commit -m "feat: POST /jobs + GET /jobs/{id} with BOLA ownership and upload validation"
```

---

### Task 3: Approve/resume, trace, and rate limiting

**Files:**
- Modify: `employee_agent/api/schemas.py` (add `ApproveRequest`, `TraceStep`, `TraceResponse`)
- Modify: `employee_agent/api/app.py` (add `POST /jobs/{id}/approve`, `GET /jobs/{id}/trace`)
- Test: `tests/api/test_approve_trace.py`, `tests/api/test_rate_limit.py`

**Interfaces:**
- `ApproveRequest(action: Literal["approve","edit","reject"]="approve", edits: dict = {})`.
- `TraceStep(step: int, node: str)`, `TraceResponse(job_id: str, steps: list[TraceStep])`.
- `POST /jobs/{id}/approve` → ownership-checked; resumes the graph with `Command(resume=body.model_dump())`; returns `JobResponse` (`done`/`error`).
- `GET /jobs/{id}/trace` → ownership-checked; returns the node execution path from checkpoint history.
- `POST /jobs` is rate-limited per key via `@limiter.limit(_limit_value)` (already decorated in Task 2); `_limit_value` reads `settings.rate_limit`.

- [ ] **Step 1: Write the failing tests**

`tests/api/test_approve_trace.py`:

```python
import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


def _app(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    prov = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    return create_app(s, provider=prov)


def _client(app):
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _files():
    return {"resume": ("resume.txt", b"Ada Lovelace. Python and Django.", "text/plain")}


async def _new_job(c):
    return (await c.post("/jobs", headers={"x-api-key": "key-a"},
                         data={"job_description": "Senior Python"}, files=_files())).json()["job_id"]


async def test_approve_finalizes_done(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.post(f"/jobs/{job_id}/approve", headers={"x-api-key": "key-a"},
                         json={"action": "approve"})
        assert r.status_code == 200
        assert r.json()["status"] == "done"
        assert r.json()["assessment"]["human_approved"] is True


async def test_reject_finalizes_error(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.post(f"/jobs/{job_id}/approve", headers={"x-api-key": "key-a"},
                         json={"action": "reject"})
        assert r.json()["status"] == "error"


async def test_trace_lists_execution_path(tmp_path):
    async with _client(_app(tmp_path)) as c:
        job_id = await _new_job(c)
        r = await c.get(f"/jobs/{job_id}/trace", headers={"x-api-key": "key-a"})
        assert r.status_code == 200
        nodes = [s["node"] for s in r.json()["steps"]]
        for expected in ["manager", "parser", "retriever", "analyst", "verifier"]:
            assert expected in nodes
```

`tests/api/test_rate_limit.py`:

```python
import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada", years_experience=1.0, top_skills=["p"], skill_matches=[],
    overall_match_score=50, recommendation="hold", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


async def test_rate_limit_returns_429(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="rl-key",
                 rate_limit="3/minute",
                 sqlite_path=str(tmp_path / "ck.sqlite"),
                 chroma_path=str(tmp_path / "chroma"))
    app = create_app(s, provider=FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT}))
    files = {"resume": ("resume.txt", b"Ada. Python.", "text/plain")}
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        codes = []
        for _ in range(4):
            r = await c.post("/jobs", headers={"x-api-key": "rl-key"},
                             data={"job_description": "x"}, files=files)
            codes.append(r.status_code)
        assert codes[:3] == [200, 200, 200]
        assert codes[3] == 429
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/api/test_approve_trace.py tests/api/test_rate_limit.py -v`
Expected: FAIL (approve/trace routes 404; `ApproveRequest` missing).

- [ ] **Step 3: Extend `employee_agent/api/schemas.py`**

Append:

```python
from typing import Literal


class ApproveRequest(BaseModel):
    action: Literal["approve", "edit", "reject"] = "approve"
    edits: dict = {}


class TraceStep(BaseModel):
    step: int
    node: str


class TraceResponse(BaseModel):
    job_id: str
    steps: list[TraceStep]
```

- [ ] **Step 4: Add the routes to `employee_agent/api/app.py`**

Extend the schema import:

```python
from employee_agent.api.schemas import (
    ApproveRequest,
    CreateJobResponse,
    JobResponse,
    TraceResponse,
    TraceStep,
)
```

Add `from langgraph.types import Command` to the imports, and add these routes (after `get_job`):

```python
@router.post("/jobs/{job_id}/approve", response_model=JobResponse)
async def approve_job(
    job_id: str, body: ApproveRequest, request: Request,
    api_key: str = Depends(require_api_key),
):
    _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = build_graph(request.app.state.provider, request.app.state.retriever, checkpointer=saver)
        result = await graph.ainvoke(Command(resume=body.model_dump()), _thread(job_id))
    status = _status_from_result(result)
    request.app.state.jobs.set_status(job_id, status)
    assessment = result.get("assessment")
    return JobResponse(
        job_id=job_id, status=status,
        assessment=assessment.model_dump() if assessment else None,
    )


@router.get("/jobs/{job_id}/trace", response_model=TraceResponse)
async def job_trace(job_id: str, request: Request, api_key: str = Depends(require_api_key)):
    _owned_or_404(request, job_id, api_key)
    async with sqlite_checkpointer(request.app.state.settings.sqlite_path) as saver:
        graph = build_graph(request.app.state.provider, request.app.state.retriever, checkpointer=saver)
        history = [snap async for snap in graph.aget_state_history(_thread(job_id))]
    steps: list[TraceStep] = []
    for snap in reversed(history):
        if snap.next:
            node = snap.next[0]
            if node != "__start__":
                steps.append(TraceStep(step=(snap.metadata or {}).get("step", 0), node=node))
    return TraceResponse(job_id=job_id, steps=steps)
```

- [ ] **Step 5: Run the API tests**

Run: `.venv/bin/python -m pytest tests/api -v`
Expected: PASS (basics 2, store 3, jobs 5, approve/trace 3, rate limit 1 = 14 passed).

> **Watch-point:** if slowapi does not pass the `request` to `_limit_value`, change it to read a module global set in `create_app` (e.g., `_RATE_LIMIT["value"] = s.rate_limit`) instead. If `assessment` in the resumed `result` is a dict rather than a `CandidateAssessment`, guard `.model_dump()` with `isinstance` (return it as-is if already a dict).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + Plan 5; ~74 passed).

- [ ] **Step 7: Commit**

```bash
git add employee_agent/api/schemas.py employee_agent/api/app.py tests/api/test_approve_trace.py tests/api/test_rate_limit.py
git commit -m "feat: approve/resume, trace, and slowapi rate limiting"
```

---

## Definition of Done (Plan 5)

- `.venv/bin/python -m pytest -q` is green across the new API tests and all prior modules.
- `POST /jobs` (multipart resume + JD) runs the graph to the human gate and returns `job_id` + `awaiting_human`; `GET /jobs/{id}` returns the draft assessment; `POST /jobs/{id}/approve` resumes to `done` (approve/edit) or `error` (reject); `GET /jobs/{id}/trace` lists the node path; `GET /health` is open; `GET /whoami` needs a key.
- Security: every non-health endpoint requires a valid `X-API-Key` (401 otherwise); a key can only read/act on its own jobs (404 otherwise); `POST /jobs` is rate-limited (429 past the per-key limit); uploads are validated by type and size.
- No network calls or model downloads in the test suite.

## Next Plan

Plan 6 — **MCP**: a mock MCP server exposing a `verify_certification` tool and an MCP client wired into the Analyst (allowlisted per `RoleConfig.tool_allowlist`), so the agent really calls an external tool. It depends on the Analyst node and the role registry.
