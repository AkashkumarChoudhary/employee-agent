# Employee Agent — Plan 3: The Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LangGraph orchestrator-worker graph — **Manager → Parser → Retriever → Analyst** over the Plan 1 `AgentState`, with a **SQLite checkpointer** for durable/resumable state and **structured output** (a Pydantic `CandidateAssessment`) from the Analyst — so the API (Plan 5) can run a resume/JD through one compiled graph and get a validated assessment.

**Architecture:** A new `employee_agent/engine/` package. Nodes are built by dependency-injecting factory functions (`make_parser(retriever)`, `make_analyst(provider)`, …) that each return an `async` callable taking the state and returning a partial-state update — so every node is unit-testable in isolation with `FakeProvider`. `graph.py` wires the four nodes into a linear `StateGraph` and compiles it with an optional checkpointer. `checkpoint.py` provides an `AsyncSqliteSaver` context manager; tests use `MemorySaver` for the fast path and one real SQLite test to prove durable persistence. `new_state()` builds a fully-populated initial `AgentState`.

**Tech Stack:** Python 3.11+, Pydantic v2, `langgraph`, `langgraph-checkpoint-sqlite` (+ `aiosqlite`), plus the Plan 1/2 modules (`providers`, `rag`, `roles`, `schemas`), pytest, pytest-asyncio.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network and never cost money** — the graph runs with `FakeProvider` (deterministic structured output + embeddings) and an ephemeral `VectorStore`; no LLM or embedding calls leave the process.
- Reuse existing interfaces verbatim:
  - `employee_agent.schemas.AgentState` (TypedDict), `CandidateAssessment`, `Chunk`, `RoleConfig`.
  - `employee_agent.providers.base.Provider` (`async generate_structured(*, system, prompt, schema)`), `employee_agent.providers.fake.FakeProvider`.
  - `employee_agent.rag.retriever.Retriever` (`async index(namespace, chunks)`, `async retrieve(namespace, query, k)`), `employee_agent.rag.store.VectorStore`, `employee_agent.rag.ingest.split_text`.
  - `employee_agent.roles.registry.get_role`.
- Node contract: each node is `async (state: AgentState) -> dict` returning only the keys it changes (LangGraph merges with last-value-wins; no custom reducers).
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`.
- Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: Engine dependencies & graph nodes

**Files:**
- Modify: `pyproject.toml` (add `langgraph`, `langgraph-checkpoint-sqlite`)
- Create: `employee_agent/engine/__init__.py`
- Create: `employee_agent/engine/nodes.py`
- Test: `tests/engine/test_nodes.py`

**Interfaces:**
- Consumes: `Provider`, `Retriever`, `ingest.split_text`, `AgentState`, `CandidateAssessment`, `RoleConfig`.
- Produces (importable from `employee_agent.engine.nodes`):
  - Constants `RESUME_CHUNK_SIZE = 800`, `RESUME_CHUNK_OVERLAP = 100`, `RETRIEVE_K = 6`.
  - `make_manager() -> node` — async node; returns `{"status": "running", "retry_count": <existing or 0>}`.
  - `make_parser(retriever: Retriever) -> node` — async node; normalizes `parsed_resume` whitespace, splits it into resume `Chunk`s, indexes them under `state["job_id"]`, returns `{"parsed_resume": <normalized>}`.
  - `make_retriever_node(retriever: Retriever) -> node` — async node; retrieves top-`RETRIEVE_K` chunks for `state["job_description"]` under `state["job_id"]`, returns `{"retrieved_chunks": [...]}`.
  - `make_analyst(provider: Provider) -> node` — async node; builds a prompt from the JD + retrieved evidence, calls `provider.generate_structured(system=role.system_prompt, prompt=..., schema=CandidateAssessment)`, returns `{"assessment": <CandidateAssessment>, "status": "done"}`.

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

Add to the `dependencies` array (after `chromadb>=0.5`):

```toml
    "langgraph>=0.2",
    "langgraph-checkpoint-sqlite>=2.0",
```

- [ ] **Step 2: Install the new dependencies**

Run: `.venv/bin/python -m pip install "langgraph>=0.2" "langgraph-checkpoint-sqlite>=2.0"`
Expected: installs `langgraph`, `langgraph-checkpoint-sqlite`, `aiosqlite`, etc.

- [ ] **Step 3: Create `employee_agent/engine/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write the failing test** in `tests/engine/test_nodes.py`

```python
from employee_agent.engine import nodes
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, Chunk

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace",
    years_experience=5.0,
    top_skills=["python", "django"],
    skill_matches=[],
    overall_match_score=80,
    recommendation="advance",
    rationale="Strong match.",
)


def _base_state(**overrides):
    state = {
        "job_id": "eng-1",
        "role_config": get_role("hr_analyst"),
        "job_description": "Senior Python engineer with Django.",
        "parsed_resume": "Ada Lovelace. 5 years Python and Django.",
        "retrieved_chunks": [],
        "assessment": None,
        "verifier_verdict": None,
        "retry_count": 0,
        "status": "running",
    }
    state.update(overrides)
    return state


async def test_manager_sets_running():
    out = await nodes.make_manager()(_base_state(status="new"))
    assert out["status"] == "running"
    assert out["retry_count"] == 0


async def test_parser_normalizes_and_indexes():
    retr = Retriever(FakeProvider(), VectorStore())
    parser = nodes.make_parser(retr)
    out = await parser(_base_state(job_id="eng-parse",
                                   parsed_resume="  Ada   Lovelace\n\nPython  Django "))
    assert out["parsed_resume"] == "Ada Lovelace Python Django"
    # side effect: resume chunks were indexed and are now retrievable
    hits = await retr.retrieve("eng-parse", "Ada Lovelace Python Django", k=3)
    assert hits


async def test_retriever_node_sets_retrieved_chunks():
    retr = Retriever(FakeProvider(), VectorStore())
    await retr.index("eng-ret", [Chunk(text="python and django", source="resume", score=0.0)])
    out = await nodes.make_retriever_node(retr)(
        _base_state(job_id="eng-ret", job_description="python and django")
    )
    assert out["retrieved_chunks"]
    assert out["retrieved_chunks"][0].text == "python and django"


async def test_analyst_produces_structured_assessment():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await nodes.make_analyst(provider)(
        _base_state(retrieved_chunks=[Chunk(text="5 yrs Python", source="resume", score=0.9)])
    )
    assert out["assessment"] == CANNED
    assert out["status"] == "done"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_nodes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.engine.nodes'`

- [ ] **Step 6: Implement `employee_agent/engine/nodes.py`**

```python
from employee_agent.providers.base import Provider
from employee_agent.rag import ingest
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState, CandidateAssessment

RESUME_CHUNK_SIZE = 800
RESUME_CHUNK_OVERLAP = 100
RETRIEVE_K = 6


def make_manager():
    async def manager(state: AgentState) -> dict:
        return {"status": "running", "retry_count": state.get("retry_count", 0)}

    return manager


def make_parser(retriever: Retriever):
    async def parser(state: AgentState) -> dict:
        text = " ".join(state["parsed_resume"].split())
        chunks = ingest.split_text(
            text, source="resume",
            chunk_size=RESUME_CHUNK_SIZE, chunk_overlap=RESUME_CHUNK_OVERLAP,
        )
        await retriever.index(state["job_id"], chunks)
        return {"parsed_resume": text}

    return parser


def make_retriever_node(retriever: Retriever):
    async def retriever_node(state: AgentState) -> dict:
        chunks = await retriever.retrieve(
            state["job_id"], query=state["job_description"], k=RETRIEVE_K
        )
        return {"retrieved_chunks": chunks}

    return retriever_node


def make_analyst(provider: Provider):
    async def analyst(state: AgentState) -> dict:
        role = state["role_config"]
        evidence = "\n".join(f"- {c.text}" for c in state["retrieved_chunks"])
        prompt = (
            f"Job description:\n{state['job_description']}\n\n"
            f"Relevant resume evidence:\n{evidence}\n\n"
            "Assess the candidate against the job description. Cite evidence; "
            "do not invent experience."
        )
        assessment = await provider.generate_structured(
            system=role.system_prompt, prompt=prompt, schema=CandidateAssessment
        )
        return {"assessment": assessment, "status": "done"}

    return analyst
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_nodes.py -v`
Expected: PASS (4 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml employee_agent/engine/__init__.py employee_agent/engine/nodes.py tests/engine/test_nodes.py
git commit -m "feat: engine graph nodes (manager, parser, retriever, analyst)"
```

---

### Task 2: Graph builder & initial state

**Files:**
- Create: `employee_agent/engine/state.py`
- Create: `employee_agent/engine/graph.py`
- Test: `tests/engine/test_graph.py`

**Interfaces:**
- Consumes: node factories from Task 1, `AgentState`, `RoleConfig`, `Provider`, `Retriever`, `langgraph`.
- Produces:
  - `employee_agent.engine.state.new_state(job_id: str, role_config: RoleConfig, job_description: str, resume_text: str) -> AgentState` — fully-populated initial state (`retrieved_chunks=[]`, `assessment=None`, `verifier_verdict=None`, `retry_count=0`, `status="running"`).
  - `employee_agent.engine.graph.build_graph(provider: Provider, retriever: Retriever, checkpointer=None)` — compiles `START → manager → parser → retriever → analyst → END`; passes `checkpointer` to `.compile()` when provided. Returns the compiled graph.

- [ ] **Step 1: Write the failing test** in `tests/engine/test_graph.py`

```python
from langgraph.checkpoint.memory import MemorySaver

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace",
    years_experience=5.0,
    top_skills=["python"],
    skill_matches=[],
    overall_match_score=88,
    recommendation="advance",
    rationale="Strong Python and Django background.",
)


def _wire():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    retriever = Retriever(provider, VectorStore())
    graph = build_graph(provider, retriever, checkpointer=MemorySaver())
    return graph


def test_new_state_is_fully_populated():
    s = new_state("j1", get_role("hr_analyst"), "JD text", "resume text")
    assert s["status"] == "running"
    assert s["retrieved_chunks"] == []
    assert s["assessment"] is None
    assert s["retry_count"] == 0


async def test_graph_runs_end_to_end_and_produces_assessment():
    graph = _wire()
    state = new_state("graph-1", get_role("hr_analyst"),
                      "Senior Python engineer with Django.",
                      "Ada Lovelace. 5 years Python and Django.")
    final = await graph.ainvoke(state, {"configurable": {"thread_id": "graph-1"}})
    assert final["status"] == "done"
    assert final["assessment"] == CANNED
    assert final["retrieved_chunks"]  # resume was indexed then retrieved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.engine.graph'`

- [ ] **Step 3: Implement `employee_agent/engine/state.py`**

```python
from employee_agent.schemas import AgentState, RoleConfig


def new_state(
    job_id: str, role_config: RoleConfig, job_description: str, resume_text: str
) -> AgentState:
    return {
        "job_id": job_id,
        "role_config": role_config,
        "job_description": job_description,
        "parsed_resume": resume_text,
        "retrieved_chunks": [],
        "assessment": None,
        "verifier_verdict": None,
        "retry_count": 0,
        "status": "running",
    }
```

- [ ] **Step 4: Implement `employee_agent/engine/graph.py`**

```python
from langgraph.graph import END, START, StateGraph

from employee_agent.engine.nodes import (
    make_analyst,
    make_manager,
    make_parser,
    make_retriever_node,
)
from employee_agent.providers.base import Provider
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState


def build_graph(provider: Provider, retriever: Retriever, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("manager", make_manager())
    g.add_node("parser", make_parser(retriever))
    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("analyst", make_analyst(provider))
    g.add_edge(START, "manager")
    g.add_edge("manager", "parser")
    g.add_edge("parser", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", END)
    return g.compile(checkpointer=checkpointer)
```

> **Watch-point (resolve during green):** confirm `from langgraph.graph import END, START, StateGraph` and `from langgraph.checkpoint.memory import MemorySaver` match the installed `langgraph`. If `.compile(checkpointer=None)` errors, call `.compile()` with no args when `checkpointer is None`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_graph.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add employee_agent/engine/state.py employee_agent/engine/graph.py tests/engine/test_graph.py
git commit -m "feat: compile langgraph orchestrator (manager->parser->retriever->analyst)"
```

---

### Task 3: SQLite checkpointer (durable, resumable state)

**Files:**
- Create: `employee_agent/engine/checkpoint.py`
- Test: `tests/engine/test_checkpoint.py`

**Interfaces:**
- Consumes: `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`, `build_graph`, `new_state`.
- Produces: `employee_agent.engine.checkpoint.sqlite_checkpointer(path: str)` — an async context manager yielding an `AsyncSqliteSaver` bound to a SQLite file at `path` (parent dirs created).

- [ ] **Step 1: Write the failing test** in `tests/engine/test_checkpoint.py`

```python
from employee_agent.engine.checkpoint import sqlite_checkpointer
from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=90, recommendation="advance",
    rationale="ok",
)


async def test_sqlite_checkpoint_persists_and_reloads(tmp_path):
    db = str(tmp_path / "ckpt.sqlite")
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    retriever = Retriever(provider, VectorStore())
    cfg = {"configurable": {"thread_id": "j-sql"}}
    state = new_state("j-sql", get_role("hr_analyst"),
                      "Senior Python engineer.", "Ada Lovelace. Python, Django.")

    async with sqlite_checkpointer(db) as saver:
        graph = build_graph(provider, retriever, checkpointer=saver)
        await graph.ainvoke(state, cfg)
        snap = await graph.aget_state(cfg)
        assert snap.values["status"] == "done"

    # Re-open a fresh saver on the same file: state persisted to disk.
    async with sqlite_checkpointer(db) as saver2:
        graph2 = build_graph(provider, retriever, checkpointer=saver2)
        snap2 = await graph2.aget_state(cfg)
        assert snap2.values["assessment"] == CANNED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.engine.checkpoint'`

- [ ] **Step 3: Implement `employee_agent/engine/checkpoint.py`**

```python
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def sqlite_checkpointer(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        yield saver
```

> **Watch-point (resolve during green):** verify the import path `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver` and that `AsyncSqliteSaver.from_conn_string(path)` is an async context manager in the installed `langgraph-checkpoint-sqlite`. Also confirm `graph.aget_state(config).values` returns the state dict (pydantic models in state must round-trip through langgraph's serializer; if `assessment` comes back as a dict rather than a `CandidateAssessment`, compare `snap2.values["assessment"] == CANNED.model_dump()` or re-validate — adjust the assertion, not the production code).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_checkpoint.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all Plan 1 + 2 + 3 tests green; ~47 passed)

- [ ] **Step 6: Commit**

```bash
git add employee_agent/engine/checkpoint.py tests/engine/test_checkpoint.py
git commit -m "feat: sqlite checkpointer for durable, resumable graph state"
```

---

## Definition of Done (Plan 3)

- `.venv/bin/python -m pytest -q` is green across engine nodes, graph, checkpoint, and all Plan 1/2 modules.
- `build_graph(FakeProvider(...), Retriever(...))` runs `manager → parser → retriever → analyst` end-to-end and yields a validated `CandidateAssessment` with `status == "done"` and non-empty `retrieved_chunks`.
- State persists to SQLite: after `ainvoke`, a fresh `AsyncSqliteSaver` on the same file re-reads the finished state for the same `thread_id`.
- No network calls and no model downloads in the test suite.

## Next Plan

Plan 4 — **Self-correction + HITL**: a Verifier node (CRAG/Self-RAG grading of grounding) with a bounded retry loop back to Retriever/Analyst via conditional edges, plus a Human-in-the-Loop `interrupt()` gate before the Finalizer. It will depend on `build_graph`, `AgentState`, `VerifierVerdict`, and the SQLite checkpointer built here (interrupts require a checkpointer).
