# Employee Agent — Plan 4: Self-Correction + HITL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-correction and a human gate to the graph — a **Verifier** node (CRAG/Self-RAG grounding check) that emits a `VerifierVerdict`, a **bounded retry loop** (conditional edges back to Retriever/Analyst, capped so it always terminates), and a **Human-in-the-Loop** `interrupt()` gate that pauses for approve/edit/reject before a **Finalizer** commits the result.

**Architecture:** Extends `employee_agent/engine/`. Four new nodes join the existing four: `verifier` (grades grounding, bumps `retry_count` on non-accept), `gate` (sets `status="awaiting_human"` — the durable marker of a pending human decision), `hitl` (calls `interrupt()`, then applies the human's decision to the assessment), and `finalizer` (sets terminal `status`). Routing after the verifier is a pure function `route_after_verifier` in a new `routing.py`, capped by `MAX_RETRIES`. The Analyst stops setting `status="done"` (the graph no longer ends there). The graph now pauses at the HITL interrupt and resumes with `Command(resume=<decision>)`.

**Tech Stack:** Python 3.11+, Pydantic v2, `langgraph` 1.x (`interrupt`, `Command`, `add_conditional_edges`), plus existing modules; pytest, pytest-asyncio.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network** — the graph runs with `FakeProvider` (fixed `CandidateAssessment` + `VerifierVerdict` per schema) and an ephemeral `VectorStore`; HITL uses `MemorySaver`.
- Reuse existing interfaces verbatim: `AgentState`, `CandidateAssessment`, `VerifierVerdict`, `Chunk`, `RoleConfig`; `Provider.generate_structured`; `FakeProvider(responses={Schema: instance})`; `Retriever`, `VectorStore`; `build_graph`, `new_state`; `get_role`.
- **Confirmed langgraph 1.2.7 behavior** (probed): dynamic `interrupt(payload)` inside a node suspends; the result of `ainvoke` then contains a `"__interrupt__"` key; `graph.aget_state(cfg).values` holds the pre-interrupt state and `.next == ("hitl",)`; resuming with `await graph.ainvoke(Command(resume=<value>), cfg)` re-runs the interrupted node with `interrupt()` returning `<value>`. Interrupts require a checkpointer.
- Node contract unchanged: `async (state: AgentState) -> dict`, returning only changed keys.
- `MAX_RETRIES = 2` (up to two self-correction loops, then escalate to the human — never infinite).
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`. Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: Verifier node & bounded retry routing

**Files:**
- Modify: `employee_agent/engine/nodes.py` (add `make_verifier`)
- Create: `employee_agent/engine/routing.py`
- Test: `tests/engine/test_verifier.py`

**Interfaces:**
- Consumes: `Provider`, `AgentState`, `VerifierVerdict`.
- Produces:
  - `nodes.make_verifier(provider: Provider) -> node` — grades whether `state["assessment"]` is grounded in `state["retrieved_chunks"]`; returns `{"verifier_verdict": <VerifierVerdict>}`, and additionally `{"retry_count": state["retry_count"] + 1}` when `verdict.action != "accept"`.
  - `routing.MAX_RETRIES = 2`.
  - `routing.route_after_verifier(state: AgentState) -> str` — returns `"gate"` if `verdict.action == "accept"` or `state["retry_count"] > MAX_RETRIES`; else `"retriever"` for `"retry_retrieval"` or `"analyst"` for `"retry_analysis"`.

- [ ] **Step 1: Write the failing test** in `tests/engine/test_verifier.py`

```python
from employee_agent.engine import nodes
from employee_agent.engine.routing import MAX_RETRIES, route_after_verifier
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, Chunk, VerifierVerdict

ASSESSMENT = CandidateAssessment(
    candidate_name="Ada", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=80, recommendation="advance", rationale="ok",
)


def _state(**overrides):
    s = {
        "job_id": "v1", "role_config": None, "job_description": "jd",
        "parsed_resume": "r", "retrieved_chunks": [Chunk(text="python", source="resume", score=0.9)],
        "assessment": ASSESSMENT, "verifier_verdict": None, "retry_count": 0,
        "status": "running",
    }
    s.update(overrides)
    return s


async def test_verifier_accept_does_not_bump_retry():
    v = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    out = await nodes.make_verifier(FakeProvider(responses={VerifierVerdict: v}))(_state())
    assert out["verifier_verdict"].action == "accept"
    assert "retry_count" not in out  # unchanged


async def test_verifier_nonaccept_bumps_retry():
    v = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis")
    out = await nodes.make_verifier(FakeProvider(responses={VerifierVerdict: v}))(
        _state(retry_count=1)
    )
    assert out["retry_count"] == 2


def test_route_accept_goes_to_gate():
    s = _state(verifier_verdict=VerifierVerdict(grounded=True, unsupported_claims=[], action="accept"))
    assert route_after_verifier(s) == "gate"


def test_route_retry_retrieval_under_cap_goes_to_retriever():
    s = _state(retry_count=1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_retrieval"))
    assert route_after_verifier(s) == "retriever"


def test_route_retry_analysis_under_cap_goes_to_analyst():
    s = _state(retry_count=1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis"))
    assert route_after_verifier(s) == "analyst"


def test_route_over_cap_escalates_to_gate():
    s = _state(retry_count=MAX_RETRIES + 1,
               verifier_verdict=VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis"))
    assert route_after_verifier(s) == "gate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_verifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.engine.routing'` (and `make_verifier` missing).

- [ ] **Step 3: Add `make_verifier` to `employee_agent/engine/nodes.py`**

Add this import at the top (alongside the existing schema import):

```python
from employee_agent.schemas import AgentState, CandidateAssessment, VerifierVerdict
```

Append this factory to the file:

```python
def make_verifier(provider: Provider):
    async def verifier(state: AgentState) -> dict:
        a = state["assessment"]
        evidence = "\n".join(f"- {c.text}" for c in state["retrieved_chunks"])
        prompt = (
            "Candidate assessment to check:\n"
            f"- recommendation: {a.recommendation}\n"
            f"- rationale: {a.rationale}\n"
            f"- top_skills: {a.top_skills}\n\n"
            f"Source evidence (retrieved resume chunks):\n{evidence}\n\n"
            "Are the assessment's claims grounded in the evidence? If not, decide "
            "whether to retry retrieval or retry analysis."
        )
        verdict = await provider.generate_structured(
            system=(
                "You are a strict grounding checker implementing CRAG/Self-RAG. "
                "Only accept claims supported by the evidence."
            ),
            prompt=prompt,
            schema=VerifierVerdict,
        )
        update = {"verifier_verdict": verdict}
        if verdict.action != "accept":
            update["retry_count"] = state["retry_count"] + 1
        return update

    return verifier
```

- [ ] **Step 4: Implement `employee_agent/engine/routing.py`**

```python
from employee_agent.schemas import AgentState

MAX_RETRIES = 2


def route_after_verifier(state: AgentState) -> str:
    verdict = state["verifier_verdict"]
    if verdict.action == "accept" or state["retry_count"] > MAX_RETRIES:
        return "gate"
    if verdict.action == "retry_retrieval":
        return "retriever"
    return "analyst"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_verifier.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add employee_agent/engine/nodes.py employee_agent/engine/routing.py tests/engine/test_verifier.py
git commit -m "feat: verifier node (CRAG/Self-RAG) + bounded retry routing"
```

---

### Task 2: HITL gate, decision, and finalizer nodes

**Files:**
- Modify: `employee_agent/engine/nodes.py` (add `make_gate`, `make_hitl`, `make_finalizer`; drop `status="done"` from `make_analyst`)
- Modify: `tests/engine/test_nodes.py` (update the analyst assertion for the status change)
- Test: `tests/engine/test_hitl_nodes.py`

**Interfaces:**
- Produces:
  - `nodes.make_gate() -> node` — returns `{"status": "awaiting_human"}` (durable pending-decision marker before the interrupt).
  - `nodes.make_hitl() -> node` — calls `interrupt({...})`; the resumed value is a decision dict `{"action": "approve"|"edit"|"reject", "edits": {...}?}`. Applies it: `approve`/`edit` → `human_approved=True` (edit also applies field updates); `reject` → `human_approved=False`. Returns `{"assessment": <updated>}`.
  - `nodes.make_finalizer() -> node` — returns `{"status": "done"}` if `assessment.human_approved` else `{"status": "error"}`.
  - `make_analyst` no longer returns `status` (returns only `{"assessment": ...}`), so the graph continues to the verifier.

- [ ] **Step 1: Write the failing test** in `tests/engine/test_hitl_nodes.py`

```python
from employee_agent.engine import nodes
from employee_agent.schemas import CandidateAssessment

ASSESSMENT = CandidateAssessment(
    candidate_name="Ada", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=70, recommendation="hold", rationale="ok",
)


async def test_gate_sets_awaiting_human():
    out = await nodes.make_gate()({"status": "running"})
    assert out["status"] == "awaiting_human"


async def test_finalizer_done_when_approved():
    approved = ASSESSMENT.model_copy(update={"human_approved": True})
    out = await nodes.make_finalizer()({"assessment": approved})
    assert out["status"] == "done"


async def test_finalizer_error_when_not_approved():
    out = await nodes.make_finalizer()({"assessment": ASSESSMENT})  # human_approved False
    assert out["status"] == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_hitl_nodes.py -v`
Expected: FAIL (`make_gate`/`make_finalizer` not defined).

- [ ] **Step 3: Update `make_analyst` in `employee_agent/engine/nodes.py`**

Change the analyst's return from `{"assessment": assessment, "status": "done"}` to:

```python
        return {"assessment": assessment}
```

- [ ] **Step 4: Add HITL node factories to `employee_agent/engine/nodes.py`**

Add this import at the top:

```python
from langgraph.types import interrupt
```

Append:

```python
def make_gate():
    async def gate(state: AgentState) -> dict:
        return {"status": "awaiting_human"}

    return gate


def make_hitl():
    async def hitl(state: AgentState) -> dict:
        decision = interrupt(
            {
                "assessment": state["assessment"].model_dump(),
                "message": "Approve, edit, or reject this candidate assessment.",
            }
        ) or {}
        action = decision.get("action", "approve")
        assessment = state["assessment"]
        if action == "edit":
            assessment = assessment.model_copy(update=decision.get("edits", {}))
            assessment = assessment.model_copy(update={"human_approved": True})
        elif action == "reject":
            assessment = assessment.model_copy(update={"human_approved": False})
        else:  # approve
            assessment = assessment.model_copy(update={"human_approved": True})
        return {"assessment": assessment}

    return hitl


def make_finalizer():
    async def finalizer(state: AgentState) -> dict:
        return {"status": "done" if state["assessment"].human_approved else "error"}

    return finalizer
```

- [ ] **Step 5: Update the analyst test** in `tests/engine/test_nodes.py`

Replace the body of `test_analyst_produces_structured_assessment` so it no longer asserts a status (the analyst stopped setting it):

```python
async def test_analyst_produces_structured_assessment():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await nodes.make_analyst(provider)(
        _base_state(retrieved_chunks=[Chunk(text="5 yrs Python", source="resume", score=0.9)])
    )
    assert out["assessment"] == CANNED
    assert "status" not in out
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/engine/test_hitl_nodes.py tests/engine/test_nodes.py -v`
Expected: PASS (3 + 4 = 7 passed)

- [ ] **Step 7: Commit**

```bash
git add employee_agent/engine/nodes.py tests/engine/test_hitl_nodes.py tests/engine/test_nodes.py
git commit -m "feat: HITL gate/decision/finalizer nodes; analyst yields to verifier"
```

---

### Task 3: Rewire the graph (conditional edges + interrupt) and update flow tests

**Files:**
- Modify: `employee_agent/engine/graph.py` (`build_graph` wires the full self-correcting + HITL flow)
- Modify: `tests/engine/test_graph.py` (end-to-end now pauses at HITL, then resumes)
- Modify: `tests/engine/test_checkpoint.py` (resume through HITL before asserting done)
- Test: `tests/engine/test_graph_hitl.py`

**Interfaces:**
- `build_graph(provider, retriever, checkpointer=None)` compiles:
  `START → manager → parser → retriever → analyst → verifier`, then a conditional edge from `verifier` via `route_after_verifier` to one of `{"retriever", "analyst", "gate"}`, then `gate → hitl → finalizer → END`. Signature unchanged.

- [ ] **Step 1: Write the failing test** in `tests/engine/test_graph_hitl.py`

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from employee_agent.engine.graph import build_graph
from employee_agent.engine.state import new_state
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="ok",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
RETRY = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="retry_analysis")


def _wire(verdict):
    provider = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: verdict})
    return build_graph(provider, Retriever(provider, VectorStore()), checkpointer=MemorySaver())


def _state(job_id):
    return new_state(job_id, get_role("hr_analyst"),
                     "Senior Python engineer.", "Ada Lovelace. Python, Django.")


async def test_accept_pauses_for_human_then_finalizes_on_approve():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-approve"}}
    res = await graph.ainvoke(_state("hitl-approve"), cfg)
    assert "__interrupt__" in res
    snap = await graph.aget_state(cfg)
    assert snap.values["status"] == "awaiting_human"
    assert snap.next == ("hitl",)
    final = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert final["status"] == "done"
    assert final["assessment"].human_approved is True


async def test_reject_finalizes_as_error():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-reject"}}
    await graph.ainvoke(_state("hitl-reject"), cfg)
    final = await graph.ainvoke(Command(resume={"action": "reject"}), cfg)
    assert final["status"] == "error"
    assert final["assessment"].human_approved is False


async def test_edit_applies_field_updates():
    graph = _wire(ACCEPT)
    cfg = {"configurable": {"thread_id": "hitl-edit"}}
    await graph.ainvoke(_state("hitl-edit"), cfg)
    final = await graph.ainvoke(
        Command(resume={"action": "edit", "edits": {"recommendation": "hold"}}), cfg
    )
    assert final["status"] == "done"
    assert final["assessment"].recommendation == "hold"
    assert final["assessment"].human_approved is True


async def test_bounded_retry_loop_terminates_at_human_gate():
    graph = _wire(RETRY)  # verifier always says "retry" -> loop must be capped
    cfg = {"configurable": {"thread_id": "hitl-loop"}}
    res = await graph.ainvoke(_state("hitl-loop"), cfg)
    assert "__interrupt__" in res  # reached the gate/interrupt, did not run forever
    snap = await graph.aget_state(cfg)
    assert snap.values["status"] == "awaiting_human"
    assert snap.values["retry_count"] >= 2  # loop ran to the cap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_graph_hitl.py -v`
Expected: FAIL — the current linear graph has no verifier/gate/hitl, so `__interrupt__` is absent and assertions fail.

- [ ] **Step 3: Rewire `build_graph` in `employee_agent/engine/graph.py`**

Replace the whole file with:

```python
from langgraph.graph import END, START, StateGraph

from employee_agent.engine.nodes import (
    make_analyst,
    make_finalizer,
    make_gate,
    make_hitl,
    make_manager,
    make_parser,
    make_retriever_node,
    make_verifier,
)
from employee_agent.engine.routing import route_after_verifier
from employee_agent.providers.base import Provider
from employee_agent.rag.retriever import Retriever
from employee_agent.schemas import AgentState


def build_graph(provider: Provider, retriever: Retriever, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("manager", make_manager())
    g.add_node("parser", make_parser(retriever))
    g.add_node("retriever", make_retriever_node(retriever))
    g.add_node("analyst", make_analyst(provider))
    g.add_node("verifier", make_verifier(provider))
    g.add_node("gate", make_gate())
    g.add_node("hitl", make_hitl())
    g.add_node("finalizer", make_finalizer())

    g.add_edge(START, "manager")
    g.add_edge("manager", "parser")
    g.add_edge("parser", "retriever")
    g.add_edge("retriever", "analyst")
    g.add_edge("analyst", "verifier")
    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"retriever": "retriever", "analyst": "analyst", "gate": "gate"},
    )
    g.add_edge("gate", "hitl")
    g.add_edge("hitl", "finalizer")
    g.add_edge("finalizer", END)
    return g.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Update `tests/engine/test_graph.py`**

Replace `test_graph_runs_end_to_end_and_produces_assessment` with the HITL-aware flow (and add the imports `from langgraph.types import Command` and `from employee_agent.schemas import CandidateAssessment, VerifierVerdict`, plus an `ACCEPT` verdict; update `_wire` to register it):

```python
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


def _wire():
    provider = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    retriever = Retriever(provider, VectorStore())
    graph = build_graph(provider, retriever, checkpointer=MemorySaver())
    return graph


async def test_graph_runs_then_pauses_for_human_and_finalizes():
    graph = _wire()
    cfg = {"configurable": {"thread_id": "graph-1"}}
    res = await graph.ainvoke(
        new_state("graph-1", get_role("hr_analyst"),
                  "Senior Python engineer with Django.",
                  "Ada Lovelace. 5 years Python and Django."),
        cfg,
    )
    assert "__interrupt__" in res  # pauses at the human gate
    assert (await graph.aget_state(cfg)).values["retrieved_chunks"]
    final = await graph.ainvoke(Command(resume={"action": "approve"}), cfg)
    assert final["status"] == "done"
    assert final["assessment"] == CANNED.model_copy(update={"human_approved": True})
```

> Keep `from langgraph.types import Command` and `from employee_agent.schemas import CandidateAssessment, VerifierVerdict` at the top of the file. `test_new_state_is_fully_populated` is unchanged.

- [ ] **Step 5: Update `tests/engine/test_checkpoint.py`**

The graph now pauses at HITL, so resume before asserting the finished state. Add `from langgraph.types import Command` and `from employee_agent.schemas import VerifierVerdict`; register an accept verdict; resume through the gate:

```python
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


async def test_sqlite_checkpoint_persists_and_reloads(tmp_path):
    db = str(tmp_path / "ckpt.sqlite")
    provider = FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT})
    retriever = Retriever(provider, VectorStore())
    cfg = {"configurable": {"thread_id": "j-sql"}}
    state = new_state("j-sql", get_role("hr_analyst"),
                      "Senior Python engineer.", "Ada Lovelace. Python, Django.")

    async with sqlite_checkpointer(db) as saver:
        graph = build_graph(provider, retriever, checkpointer=saver)
        await graph.ainvoke(state, cfg)                    # pauses at HITL
        await graph.ainvoke(Command(resume={"action": "approve"}), cfg)  # finish
        assert (await graph.aget_state(cfg)).values["status"] == "done"

    # Re-open a fresh saver on the same file: finished state persisted to disk.
    async with sqlite_checkpointer(db) as saver2:
        graph2 = build_graph(provider, retriever, checkpointer=saver2)
        snap2 = await graph2.aget_state(cfg)
        assert snap2.values["assessment"].human_approved is True
        assert snap2.values["status"] == "done"
```

- [ ] **Step 6: Run the affected engine tests**

Run: `.venv/bin/python -m pytest tests/engine -v`
Expected: PASS (nodes 4, verifier 6, hitl_nodes 3, graph 2, graph_hitl 4, checkpoint 1 = 20 passed).

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + Plan 4; ~60 passed).

- [ ] **Step 8: Commit**

```bash
git add employee_agent/engine/graph.py tests/engine/test_graph.py tests/engine/test_checkpoint.py tests/engine/test_graph_hitl.py
git commit -m "feat: self-correcting graph with conditional retry loop and HITL interrupt"
```

---

## Definition of Done (Plan 4)

- `.venv/bin/python -m pytest -q` is green across the new verifier/HITL tests and all prior modules.
- The graph pauses at the HITL `interrupt()` (`"__interrupt__"` present, `status == "awaiting_human"`, `next == ("hitl",)`), and resumes via `Command(resume={"action": ...})`: **approve** → `status="done"` + `human_approved=True`; **edit** → fields updated + done; **reject** → `status="error"` + `human_approved=False`.
- The verifier's retry loop is bounded by `MAX_RETRIES` and always terminates at the human gate even when the verifier never accepts.
- State (including the resumed decision) round-trips through the SQLite checkpointer.
- No network calls or model downloads in the test suite.

## Next Plan

Plan 5 — **The API**: a FastAPI app wrapping the compiled graph — `POST /jobs` (resume upload + JD → starts the graph, returns `job_id`), `GET /jobs/{id}`, `POST /jobs/{id}/approve` (submits the HITL decision, resumes via `Command`), `GET /jobs/{id}/trace`, `GET /health` — with `X-API-Key` auth, per-key ownership (BOLA) checks, slowapi rate limiting, and Pydantic request/response contracts. It depends on `build_graph`, `sqlite_checkpointer`, `new_state`, `get_role`, and `ingest.load_document`.
