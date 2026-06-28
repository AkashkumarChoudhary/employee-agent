# Employee Agent — Plan 1: Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational layer — config, Pydantic data contracts, an LLM/embeddings provider abstraction with Gemini primary + Ollama failover + a test fake, and the HR role preset — so later subsystems (RAG, graph, API) have stable, tested interfaces to build on.

**Architecture:** A small `employee_agent` Python package. All data contracts live in `schemas.py`. Providers implement a single `Provider` protocol (`generate_structured`, `embed`) so the rest of the system never imports an SDK directly; a `FailoverProvider` wraps a primary + fallback, and a `FakeProvider` makes everything testable offline. Roles are declarative `RoleConfig` objects.

**Tech Stack:** Python 3.11+, Pydantic v2, pydantic-settings, google-genai (Gemini), ollama, pytest.

## Global Constraints

- Python **3.11+** (uses `X | None`, `Literal`, `TypedDict`).
- Pydantic **v2** (`BaseModel`, `field_validator`, `pydantic-settings`).
- LLM provider: **Google Gemini** free tier (`google-genai` SDK), model `gemini-2.0-flash`; embeddings model `text-embedding-004`. Failover: **Ollama** (local).
- All LLM calls use **`temperature=0`** for determinism.
- **Tests never hit the network and never cost money** — use `FakeProvider`; mock SDKs where a real provider class is under test.
- **Git is not yet initialized** in this repo (per user). The `Commit` step in each task is therefore **optional**: either run `git init` once before starting, or skip the commit steps and proceed. Commit messages are provided for when git is enabled.
- Package root: `employee_agent/`. Tests root: `tests/`.

---

### Task 1: Project skeleton & configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `employee_agent/__init__.py`
- Create: `employee_agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `employee_agent.config.Settings` (pydantic-settings model) and `get_settings() -> Settings`. Fields: `gemini_api_key: str | None`, `gemini_model: str = "gemini-2.0-flash"`, `embedding_model: str = "text-embedding-004"`, `ollama_model: str = "llama3.1"`, `ollama_host: str = "http://localhost:11434"`, `provider: Literal["gemini", "ollama", "fake"] = "gemini"`, `enable_failover: bool = True`, `temperature: float = 0.0`, `sqlite_path: str = "./data/employee_agent.db"`, `chroma_path: str = "./data/chroma"`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "employee-agent"
version = "0.1.0"
description = "Configurable digital-employee agent engine (HR analyst hero use case)"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "google-genai>=0.3",
    "ollama>=0.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["."]
```

> `pythonpath = ["."]` lets `import employee_agent` work in tests without an
> editable install. Install dev deps with `pip install pydantic pydantic-settings
> google-genai ollama pytest pytest-asyncio` (a venv is recommended).

- [ ] **Step 2: Create `.env.example`**

```bash
# Copy to .env and fill in. .env is gitignored.
GEMINI_API_KEY=your-ai-studio-key-here
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=text-embedding-004
OLLAMA_MODEL=llama3.1
OLLAMA_HOST=http://localhost:11434
PROVIDER=gemini
ENABLE_FAILOVER=true
TEMPERATURE=0.0
SQLITE_PATH=./data/employee_agent.db
CHROMA_PATH=./data/chroma
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.env
.venv/
data/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Create empty `employee_agent/__init__.py`**

```python
"""Employee Agent — configurable digital-employee engine."""
```

- [ ] **Step 5: Write the failing test** in `tests/test_config.py`

```python
from employee_agent.config import Settings, get_settings


def test_defaults_apply_when_env_absent(monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("TEMPERATURE", raising=False)
    s = Settings(_env_file=None)
    assert s.provider == "gemini"
    assert s.gemini_model == "gemini-2.0-flash"
    assert s.temperature == 0.0
    assert s.enable_failover is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PROVIDER", "fake")
    monkeypatch.setenv("TEMPERATURE", "0.7")
    s = Settings(_env_file=None)
    assert s.provider == "fake"
    assert s.temperature == 0.7


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.config'`

- [ ] **Step 7: Implement `employee_agent/config.py`**

```python
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-004"
    ollama_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    provider: Literal["gemini", "ollama", "fake"] = "gemini"
    enable_failover: bool = True
    temperature: float = 0.0
    sqlite_path: str = "./data/employee_agent.db"
    chroma_path: str = "./data/chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 9: Commit** *(optional — see Global Constraints)*

```bash
git add pyproject.toml .env.example .gitignore employee_agent/ tests/test_config.py
git commit -m "feat: project skeleton and settings config"
```

---

### Task 2: Core Pydantic schemas

**Files:**
- Create: `employee_agent/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces (importable from `employee_agent.schemas`):
  - `Chunk(text: str, source: str, score: float)`
  - `SkillMatch(requirement: str, candidate_evidence: str | None, met: bool, confidence: float)`
  - `CandidateAssessment(candidate_name, years_experience: float, top_skills: list[str], skill_matches: list[SkillMatch], overall_match_score: int, recommendation: Literal["advance","hold","reject"], rationale: str, human_approved: bool = False)`
  - `VerifierVerdict(grounded: bool, unsupported_claims: list[str], action: Literal["accept","retry_retrieval","retry_analysis"])`
  - `RoleConfig(name: str, system_prompt: str, extraction_schema: str, tool_allowlist: list[str], knowledge_namespace: str)`
  - `CreateJobRequest(role: str = "hr_analyst", job_description: str)`
  - `AgentState` (TypedDict) with keys: `job_id, role_config, job_description, parsed_resume, retrieved_chunks, assessment, verifier_verdict, retry_count, status`.

- [ ] **Step 1: Write the failing test** in `tests/test_schemas.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.schemas'`

- [ ] **Step 3: Implement `employee_agent/schemas.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit** *(optional)*

```bash
git add employee_agent/schemas.py tests/test_schemas.py
git commit -m "feat: core pydantic data contracts"
```

---

### Task 3: Provider protocol & FakeProvider

**Files:**
- Create: `employee_agent/providers/__init__.py`
- Create: `employee_agent/providers/base.py`
- Create: `employee_agent/providers/fake.py`
- Test: `tests/providers/test_fake.py`

**Interfaces:**
- Consumes: `employee_agent.schemas` (for typed structured output target).
- Produces:
  - `employee_agent.providers.base.Provider` — a `typing.Protocol` with:
    - `async def generate_structured(self, *, system: str, prompt: str, schema: type[BaseModelT]) -> BaseModelT`
    - `async def embed(self, texts: list[str]) -> list[list[float]]`
    - `name: str` (attribute)
  - `employee_agent.providers.base.ProviderError(Exception)` — raised by providers on failure.
  - `employee_agent.providers.fake.FakeProvider(responses: dict[type, BaseModel] | None = None, embed_dim: int = 8, fail: bool = False)` implementing `Provider`. `generate_structured` returns the registered instance for `schema` (or constructs a deterministic default); `embed` returns deterministic vectors derived from text hashes. If `fail=True`, both raise `ProviderError`.

- [ ] **Step 1: Write the failing test** in `tests/providers/test_fake.py`

```python
import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import VerifierVerdict


async def test_generate_structured_returns_registered_instance():
    want = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    p = FakeProvider(responses={VerifierVerdict: want})
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_embed_is_deterministic_and_right_shape():
    p = FakeProvider(embed_dim=8)
    a = await p.embed(["hello", "world"])
    b = await p.embed(["hello", "world"])
    assert len(a) == 2 and len(a[0]) == 8
    assert a == b  # deterministic


async def test_fail_mode_raises():
    p = FakeProvider(fail=True)
    with pytest.raises(ProviderError):
        await p.embed(["x"])
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
```

- [ ] **Step 2: Create `employee_agent/providers/__init__.py`** (empty)

```python
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/providers/test_fake.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.providers.base'`

- [ ] **Step 4: Implement `employee_agent/providers/base.py`**

```python
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class ProviderError(Exception):
    """Raised when a provider call fails (network, quota, parse)."""


@runtime_checkable
class Provider(Protocol):
    name: str

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 5: Implement `employee_agent/providers/fake.py`**

```python
import hashlib

from pydantic import BaseModel

from employee_agent.providers.base import BaseModelT, ProviderError


def _default_instance(schema: type[BaseModel]) -> BaseModel:
    """Construct a schema instance from each field's type default."""
    values = {}
    for fname, field in schema.model_fields.items():
        ann = field.annotation
        if ann in (int, float):
            values[fname] = 0
        elif ann is bool:
            values[fname] = False
        elif ann is str:
            values[fname] = ""
        elif ann is list or getattr(ann, "__origin__", None) is list:
            values[fname] = []
        else:
            values[fname] = None
    return schema.model_construct(**values)


class FakeProvider:
    name = "fake"

    def __init__(self, responses=None, embed_dim: int = 8, fail: bool = False):
        self._responses = responses or {}
        self._embed_dim = embed_dim
        self._fail = fail

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        if self._fail:
            raise ProviderError("fake failure")
        if schema in self._responses:
            return self._responses[schema]
        return _default_instance(schema)  # type: ignore[return-value]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise ProviderError("fake failure")
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self._embed_dim)])
        return out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/providers/test_fake.py -v`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit** *(optional)*

```bash
git add employee_agent/providers/ tests/providers/test_fake.py
git commit -m "feat: provider protocol and test fake"
```

---

### Task 4: Failover provider wrapper

**Files:**
- Create: `employee_agent/providers/failover.py`
- Test: `tests/providers/test_failover.py`

**Interfaces:**
- Consumes: `Provider`, `ProviderError` from `employee_agent.providers.base`.
- Produces: `employee_agent.providers.failover.FailoverProvider(primary: Provider, fallback: Provider)` implementing `Provider`. On `ProviderError` from `primary`, retries once on `fallback`; if both fail, re-raises the fallback's `ProviderError`. `name` = `f"{primary.name}->{fallback.name}"`.

- [ ] **Step 1: Write the failing test** in `tests/providers/test_failover.py`

```python
import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import VerifierVerdict


async def test_uses_primary_when_healthy():
    want = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    p = FailoverProvider(FakeProvider(responses={VerifierVerdict: want}),
                         FakeProvider(fail=True))
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_falls_back_when_primary_fails():
    want = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="accept")
    p = FailoverProvider(FakeProvider(fail=True),
                         FakeProvider(responses={VerifierVerdict: want}))
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_raises_when_both_fail():
    p = FailoverProvider(FakeProvider(fail=True), FakeProvider(fail=True))
    with pytest.raises(ProviderError):
        await p.embed(["x"])


async def test_name_composes():
    p = FailoverProvider(FakeProvider(), FakeProvider())
    assert p.name == "fake->fake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_failover.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.providers.failover'`

- [ ] **Step 3: Implement `employee_agent/providers/failover.py`**

```python
from employee_agent.providers.base import BaseModelT, Provider, ProviderError


class FailoverProvider:
    def __init__(self, primary: Provider, fallback: Provider):
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            return await self._primary.generate_structured(
                system=system, prompt=prompt, schema=schema
            )
        except ProviderError:
            return await self._fallback.generate_structured(
                system=system, prompt=prompt, schema=schema
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._primary.embed(texts)
        except ProviderError:
            return await self._fallback.embed(texts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_failover.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit** *(optional)*

```bash
git add employee_agent/providers/failover.py tests/providers/test_failover.py
git commit -m "feat: failover provider wrapper"
```

---

### Task 5: Gemini provider

**Files:**
- Create: `employee_agent/providers/gemini.py`
- Test: `tests/providers/test_gemini.py`

**Interfaces:**
- Consumes: `config.get_settings`, `Provider`, `ProviderError`, schemas.
- Produces: `employee_agent.providers.gemini.GeminiProvider(api_key: str, model: str, embedding_model: str, temperature: float = 0.0)` implementing `Provider`, `name = "gemini"`. Wraps `google.genai.Client`. `generate_structured` requests JSON conforming to `schema` (uses `response_schema`/`response_mime_type`) and parses into the model; any SDK exception or parse failure is re-raised as `ProviderError`. `embed` calls the embeddings endpoint.

> Test approach: the `google.genai` SDK is mocked — we assert our wrapper maps responses/errors correctly, never hitting the network.

- [ ] **Step 1: Write the failing test** in `tests/providers/test_gemini.py`

```python
from unittest.mock import MagicMock

import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.gemini import GeminiProvider
from employee_agent.schemas import VerifierVerdict


def _provider_with_mock_client(monkeypatch, client):
    monkeypatch.setattr(
        "employee_agent.providers.gemini.genai.Client",
        lambda **kw: client,
    )
    return GeminiProvider(api_key="k", model="gemini-2.0-flash",
                          embedding_model="text-embedding-004")


async def test_generate_structured_parses_json(monkeypatch):
    client = MagicMock()
    resp = MagicMock()
    resp.text = ('{"grounded": true, "unsupported_claims": [], "action": "accept"}')
    client.aio.models.generate_content = _async_return(resp)
    p = _provider_with_mock_client(monkeypatch, client)
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert isinstance(got, VerifierVerdict) and got.grounded is True


async def test_sdk_error_becomes_provider_error(monkeypatch):
    client = MagicMock()
    client.aio.models.generate_content = _async_raise(RuntimeError("429"))
    p = _provider_with_mock_client(monkeypatch, client)
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)


async def test_embed_maps_values(monkeypatch):
    client = MagicMock()
    emb = MagicMock()
    emb.embeddings = [MagicMock(values=[0.1, 0.2]), MagicMock(values=[0.3, 0.4])]
    client.aio.models.embed_content = _async_return(emb)
    p = _provider_with_mock_client(monkeypatch, client)
    out = await p.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]


# --- async helpers ---
def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f


def _async_raise(exc):
    async def _f(*a, **k):
        raise exc
    return _f
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_gemini.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.providers.gemini'`

- [ ] **Step 3: Implement `employee_agent/providers/gemini.py`**

```python
from google import genai
from google.genai import types

from employee_agent.providers.base import BaseModelT, ProviderError


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, embedding_model: str,
                 temperature: float = 0.0):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model
        self._temperature = temperature

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return schema.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001 - normalize all SDK/parse errors
            raise ProviderError(f"gemini generate failed: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.aio.models.embed_content(
                model=self._embedding_model, contents=texts
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"gemini embed failed: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_gemini.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** *(optional)*

```bash
git add employee_agent/providers/gemini.py tests/providers/test_gemini.py
git commit -m "feat: gemini provider"
```

---

### Task 6: Ollama provider

**Files:**
- Create: `employee_agent/providers/ollama_provider.py`
- Test: `tests/providers/test_ollama.py`

**Interfaces:**
- Consumes: `Provider`, `ProviderError`, schemas.
- Produces: `employee_agent.providers.ollama_provider.OllamaProvider(model: str, host: str, embedding_model: str | None = None, temperature: float = 0.0)` implementing `Provider`, `name = "ollama"`. Uses `ollama.AsyncClient`. `generate_structured` passes `format=schema.model_json_schema()` and parses the response into the model; errors → `ProviderError`. `embed` uses `client.embed`.

> Test approach: `ollama.AsyncClient` is mocked.

- [ ] **Step 1: Write the failing test** in `tests/providers/test_ollama.py`

```python
from unittest.mock import MagicMock

import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.ollama_provider import OllamaProvider
from employee_agent.schemas import VerifierVerdict


def _provider(monkeypatch, client):
    monkeypatch.setattr(
        "employee_agent.providers.ollama_provider.ollama.AsyncClient",
        lambda **kw: client,
    )
    return OllamaProvider(model="llama3.1", host="http://localhost:11434")


async def test_generate_structured_parses(monkeypatch):
    client = MagicMock()
    client.chat = _async_return(
        {"message": {"content":
         '{"grounded": false, "unsupported_claims": ["x"], "action": "retry_analysis"}'}}
    )
    p = _provider(monkeypatch, client)
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got.action == "retry_analysis"


async def test_error_becomes_provider_error(monkeypatch):
    client = MagicMock()
    client.chat = _async_raise(RuntimeError("conn refused"))
    p = _provider(monkeypatch, client)
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)


async def test_embed(monkeypatch):
    client = MagicMock()
    client.embed = _async_return({"embeddings": [[0.5, 0.6]]})
    p = _provider(monkeypatch, client)
    out = await p.embed(["a"])
    assert out == [[0.5, 0.6]]


def _async_return(value):
    async def _f(*a, **k):
        return value
    return _f


def _async_raise(exc):
    async def _f(*a, **k):
        raise exc
    return _f
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/providers/test_ollama.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `employee_agent/providers/ollama_provider.py`**

```python
import ollama

from employee_agent.providers.base import BaseModelT, ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, embedding_model: str | None = None,
                 temperature: float = 0.0):
        self._client = ollama.AsyncClient(host=host)
        self._model = model
        self._embedding_model = embedding_model or model
        self._temperature = temperature

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            resp = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format=schema.model_json_schema(),
                options={"temperature": self._temperature},
            )
            return schema.model_validate_json(resp["message"]["content"])
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"ollama generate failed: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.embed(model=self._embedding_model, input=texts)
            return [list(v) for v in resp["embeddings"]]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"ollama embed failed: {e}") from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/providers/test_ollama.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit** *(optional)*

```bash
git add employee_agent/providers/ollama_provider.py tests/providers/test_ollama.py
git commit -m "feat: ollama provider"
```

---

### Task 7: Provider factory & HR role preset

**Files:**
- Create: `employee_agent/providers/factory.py`
- Create: `employee_agent/roles/__init__.py`
- Create: `employee_agent/roles/registry.py`
- Test: `tests/providers/test_factory.py`
- Test: `tests/roles/test_registry.py`

**Interfaces:**
- Consumes: `Settings`/`get_settings`, all provider classes, `RoleConfig`.
- Produces:
  - `employee_agent.providers.factory.build_provider(settings: Settings | None = None) -> Provider` — selects provider by `settings.provider`; when `settings.enable_failover` and primary is `gemini`, wraps it in `FailoverProvider(gemini, ollama)`. For `provider="fake"` returns a `FakeProvider`. Raises `ValueError` if `provider="gemini"` and `gemini_api_key` is missing.
  - `employee_agent.roles.registry.ROLES: dict[str, RoleConfig]` and `get_role(name: str) -> RoleConfig` (raises `KeyError` for unknown). Includes the built `hr_analyst` preset.

- [ ] **Step 1: Write the failing test** in `tests/providers/test_factory.py`

```python
import pytest

from employee_agent.config import Settings
from employee_agent.providers.factory import build_provider
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider


def test_fake_provider_selected():
    p = build_provider(Settings(_env_file=None, provider="fake"))
    assert isinstance(p, FakeProvider)


def test_gemini_without_key_raises():
    s = Settings(_env_file=None, provider="gemini", gemini_api_key=None)
    with pytest.raises(ValueError):
        build_provider(s)


def test_gemini_with_failover_wraps(monkeypatch):
    monkeypatch.setattr("employee_agent.providers.factory.GeminiProvider",
                        lambda **kw: FakeProvider())
    monkeypatch.setattr("employee_agent.providers.factory.OllamaProvider",
                        lambda **kw: FakeProvider())
    s = Settings(_env_file=None, provider="gemini", gemini_api_key="k",
                 enable_failover=True)
    p = build_provider(s)
    assert isinstance(p, FailoverProvider)
```

- [ ] **Step 2: Write the failing test** in `tests/roles/test_registry.py`

```python
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
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `pytest tests/providers/test_factory.py tests/roles/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError` for `factory` / `roles.registry`

- [ ] **Step 4: Create `employee_agent/roles/__init__.py`** (empty)

```python
```

- [ ] **Step 5: Implement `employee_agent/roles/registry.py`**

```python
from employee_agent.schemas import RoleConfig

HR_ANALYST = RoleConfig(
    name="hr_analyst",
    system_prompt=(
        "You are an experienced HR recruiting analyst. Given a candidate's resume "
        "and a job description, assess fit objectively. Cite evidence from the resume "
        "for each requirement. Never invent experience that is not present. Output "
        "strictly conforms to the CandidateAssessment schema."
    ),
    extraction_schema="CandidateAssessment",
    tool_allowlist=["verify_certification"],
    knowledge_namespace="hr",
)

# Future roles (research/knowledge/support) register here as stubs in later plans.
ROLES: dict[str, RoleConfig] = {
    HR_ANALYST.name: HR_ANALYST,
}


def get_role(name: str) -> RoleConfig:
    return ROLES[name]
```

- [ ] **Step 6: Implement `employee_agent/providers/factory.py`**

```python
from employee_agent.config import Settings, get_settings
from employee_agent.providers.base import Provider
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider
from employee_agent.providers.gemini import GeminiProvider
from employee_agent.providers.ollama_provider import OllamaProvider


def build_provider(settings: Settings | None = None) -> Provider:
    s = settings or get_settings()

    if s.provider == "fake":
        return FakeProvider()

    if s.provider == "ollama":
        return OllamaProvider(model=s.ollama_model, host=s.ollama_host,
                              temperature=s.temperature)

    if s.provider == "gemini":
        if not s.gemini_api_key:
            raise ValueError("gemini_api_key is required when provider='gemini'")
        primary = GeminiProvider(
            api_key=s.gemini_api_key, model=s.gemini_model,
            embedding_model=s.embedding_model, temperature=s.temperature,
        )
        if s.enable_failover:
            fallback = OllamaProvider(model=s.ollama_model, host=s.ollama_host,
                                      temperature=s.temperature)
            return FailoverProvider(primary, fallback)
        return primary

    raise ValueError(f"unknown provider: {s.provider}")
```

- [ ] **Step 7: Run both tests to verify they pass**

Run: `pytest tests/providers/test_factory.py tests/roles/test_registry.py -v`
Expected: PASS (6 passed)

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: PASS (all tasks' tests green)

- [ ] **Step 9: Commit** *(optional)*

```bash
git add employee_agent/providers/factory.py employee_agent/roles/ tests/providers/test_factory.py tests/roles/test_registry.py
git commit -m "feat: provider factory and HR role preset"
```

---

## Definition of Done (Plan 1)

- `pytest -v` is green (config, schemas, fake/failover/gemini/ollama providers, factory, role registry).
- `build_provider()` returns a working provider for `fake`, `ollama`, and `gemini` (with failover) settings.
- `get_role("hr_analyst")` returns the HR `RoleConfig`.
- No network calls in the test suite.

## Next Plan
Plan 2 — **RAG core**: document ingestion (LangChain loaders/splitters), Chroma vector store namespaced per job, and a retriever that consumes `build_provider().embed`. It will depend on `Chunk`, `Provider`, and `get_settings` produced here.
```
