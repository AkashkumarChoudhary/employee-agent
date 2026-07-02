# Employee Agent — Plan 6: MCP Tool Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent really call an external tool over the Model Context Protocol — a mock **MCP server** exposing `verify_certification`, an allowlist-aware **MCP client**, and the **Analyst** node calling it (gated by `RoleConfig.tool_allowlist`) so a verification note flows through the graph and out of the API, resiliently (a bad tool response never breaks a job).

**Architecture:** A new `employee_agent/mcp_tools/` package (named to avoid clashing with the installed `mcp` SDK). `server.py` builds a `FastMCP` server with one tool. `client.py`'s `MCPToolClient` connects to that server **in-memory** (`mcp.shared.memory.create_connected_server_and_client_session` — no subprocess, offline, fast), enforces an allowlist, and returns the tool's dict. `make_analyst` gains an optional `tool_client`; when the role allowlists `verify_certification` it calls the tool and appends a verification note to the assessment rationale, catching any error (resilience). `build_graph` and `create_app` thread the client through.

**Tech Stack:** Python 3.11+, `mcp` 1.x (`FastMCP`, in-memory client session), plus all prior modules; pytest, pytest-asyncio.

## Global Constraints

- Python **3.11+**; Pydantic **v2**.
- **Tests never hit the network** — the MCP server/client run **in-memory** in-process; `FakeProvider` supplies the assessment. No subprocess, no sockets.
- **Confirmed (probed) with mcp 1.28.1:** `create_connected_server_and_client_session(fastmcp_server)` is an async context manager yielding a `ClientSession`; call `await session.initialize()` then `await session.call_tool(name, args)`; the result is a `CallToolResult` whose `.isError` is a bool and whose `.content[0].text` is the tool's return **dict serialized as JSON** (`.structuredContent` is `None` for a plain `-> dict` tool, so parse `content[0].text`). An MCP call nested inside a langgraph node inside `ainvoke` works.
- Reuse verbatim: `make_analyst` (extended), `build_graph` (extended), `create_app` (extended), `RoleConfig.tool_allowlist` (HR preset already lists `verify_certification`), `CandidateAssessment`, `FakeProvider`.
- **Backward compatibility:** `make_analyst(provider)` and `build_graph(provider, retriever, checkpointer=...)` must keep working (new `tool_client` param defaults to `None`), so all Plan 1–5 tests stay green unchanged.
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`. Work continues on branch `feat/employee-agent-foundations`; commit per task.

---

### Task 1: Mock MCP server

**Files:**
- Modify: `pyproject.toml` (add `mcp>=1.0`)
- Create: `employee_agent/mcp_tools/__init__.py`
- Create: `employee_agent/mcp_tools/server.py`
- Test: `tests/mcp_tools/test_server.py`

**Interfaces:**
- `mcp_tools.server.build_mcp_server() -> FastMCP` — a `FastMCP("employee-agent-tools")` with a `verify_certification(name: str, certification: str) -> dict` tool returning `{"name", "certification", "verified": bool, "source": "mock-registry"}`; `verified` is `True` iff the certification (lowercased/stripped) is in a small known set.

- [ ] **Step 1: Add dependency to `pyproject.toml`**

Add to the `dependencies` array: `"mcp>=1.0",`

- [ ] **Step 2: Install**

Run: `.venv/bin/python -m pip install "mcp>=1.0"`
Expected: `mcp` already/now installed.

- [ ] **Step 3: Create `employee_agent/mcp_tools/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write the failing test** in `tests/mcp_tools/test_server.py`

```python
import json

from mcp.shared.memory import create_connected_server_and_client_session

from employee_agent.mcp_tools.server import build_mcp_server


async def _call(cert: str) -> dict:
    server = build_mcp_server()
    async with create_connected_server_and_client_session(server) as session:
        await session.initialize()
        res = await session.call_tool(
            "verify_certification", {"name": "Ada", "certification": cert}
        )
    assert res.isError is False
    return json.loads(res.content[0].text)


async def test_known_certification_verified():
    out = await _call("PMP")
    assert out["verified"] is True
    assert out["source"] == "mock-registry"


async def test_unknown_certification_not_verified():
    out = await _call("underwater basket weaving")
    assert out["verified"] is False
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mcp_tools/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.mcp_tools.server'`

- [ ] **Step 6: Implement `employee_agent/mcp_tools/server.py`**

```python
from mcp.server.fastmcp import FastMCP

_KNOWN_CERTIFICATIONS = {
    "pmp", "cfa", "cissp", "aws certified solutions architect",
    "pmp certification", "python", "scrum master",
}


def build_mcp_server() -> FastMCP:
    server = FastMCP("employee-agent-tools")

    @server.tool()
    def verify_certification(name: str, certification: str) -> dict:
        """Verify a candidate certification against a mock registry."""
        verified = certification.strip().lower() in _KNOWN_CERTIFICATIONS
        return {
            "name": name,
            "certification": certification,
            "verified": verified,
            "source": "mock-registry",
        }

    return server
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/mcp_tools/test_server.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml employee_agent/mcp_tools/__init__.py employee_agent/mcp_tools/server.py tests/mcp_tools/test_server.py
git commit -m "feat: mock MCP server with verify_certification tool"
```

---

### Task 2: Allowlist-aware MCP client

**Files:**
- Create: `employee_agent/mcp_tools/client.py`
- Test: `tests/mcp_tools/test_client.py`

**Interfaces:**
- `mcp_tools.client.MCPToolError(Exception)`.
- `mcp_tools.client.MCPToolClient(server: FastMCP | None = None, allowlist: set[str] | None = None)`:
  - `async call(self, tool: str, args: dict) -> dict` — raises `MCPToolError` if `tool` not in `allowlist`; otherwise connects to the server in-memory, initializes, calls the tool, and returns the parsed dict; raises `MCPToolError` if the tool result `isError`.

- [ ] **Step 1: Write the failing test** in `tests/mcp_tools/test_client.py`

```python
import pytest

from employee_agent.mcp_tools.client import MCPToolClient, MCPToolError


async def test_call_allowlisted_tool_returns_dict():
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await client.call("verify_certification", {"name": "Ada", "certification": "CISSP"})
    assert out["verified"] is True
    assert out["name"] == "Ada"


async def test_unknown_cert_returns_not_verified():
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await client.call("verify_certification", {"name": "Ada", "certification": "nope"})
    assert out["verified"] is False


async def test_non_allowlisted_tool_raises():
    client = MCPToolClient(allowlist=set())
    with pytest.raises(MCPToolError):
        await client.call("verify_certification", {"name": "Ada", "certification": "PMP"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/mcp_tools/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.mcp_tools.client'`

- [ ] **Step 3: Implement `employee_agent/mcp_tools/client.py`**

```python
import json

from mcp.shared.memory import create_connected_server_and_client_session

from employee_agent.mcp_tools.server import build_mcp_server


class MCPToolError(Exception):
    """Raised when an MCP tool call is disallowed or fails."""


class MCPToolClient:
    def __init__(self, server=None, allowlist: set[str] | None = None):
        self._server = server or build_mcp_server()
        self._allowlist = set(allowlist or [])

    async def call(self, tool: str, args: dict) -> dict:
        if tool not in self._allowlist:
            raise MCPToolError(f"tool not allowlisted: {tool}")
        async with create_connected_server_and_client_session(self._server) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
        if result.isError:
            raise MCPToolError(f"tool {tool} returned an error")
        return json.loads(result.content[0].text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/mcp_tools/test_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add employee_agent/mcp_tools/client.py tests/mcp_tools/test_client.py
git commit -m "feat: allowlist-aware in-memory MCP client"
```

---

### Task 3: Wire the tool into the Analyst, graph, and API

**Files:**
- Modify: `employee_agent/engine/nodes.py` (`make_analyst` gains `tool_client=None`)
- Modify: `employee_agent/engine/graph.py` (`build_graph` gains `tool_client=None`, passes to analyst)
- Modify: `employee_agent/api/app.py` (build an `MCPToolClient`; pass it to `build_graph`)
- Test: `tests/engine/test_analyst_tool.py`, `tests/api/test_mcp_integration.py`

**Interfaces:**
- `make_analyst(provider: Provider, tool_client=None)` — after producing the assessment, if `tool_client is not None` and `"verify_certification" in role.tool_allowlist`, call the tool with `{"name": assessment.candidate_name, "certification": <top skill or "n/a">}` and append `" [MCP verify_certification: <cert> -> verified=<bool>]"` to `rationale`. Any exception is swallowed (the job continues without the tool result).
- `build_graph(provider, retriever, checkpointer=None, tool_client=None)` — passes `tool_client` to `make_analyst`.
- `create_app` sets `app.state.tool_client = MCPToolClient(allowlist={"verify_certification"})` and passes it wherever it builds the graph.

- [ ] **Step 1: Write the failing tests**

`tests/engine/test_analyst_tool.py`:

```python
from employee_agent.engine.nodes import make_analyst
from employee_agent.mcp_tools.client import MCPToolClient
from employee_agent.providers.fake import FakeProvider
from employee_agent.roles.registry import get_role
from employee_agent.schemas import CandidateAssessment, Chunk

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=80, recommendation="advance", rationale="Base.",
)


def _state(**overrides):
    s = {
        "job_id": "tool-1", "role_config": get_role("hr_analyst"),
        "job_description": "jd", "parsed_resume": "r",
        "retrieved_chunks": [Chunk(text="python", source="resume", score=0.9)],
        "assessment": None, "verifier_verdict": None, "retry_count": 0, "status": "running",
    }
    s.update(overrides)
    return s


class _BoomClient:
    async def call(self, tool, args):
        raise RuntimeError("mcp down")


async def test_analyst_appends_mcp_verification_when_allowlisted():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    client = MCPToolClient(allowlist={"verify_certification"})
    out = await make_analyst(provider, tool_client=client)(_state())
    assert "verify_certification" in out["assessment"].rationale
    assert "verified=True" in out["assessment"].rationale  # "python" is a known cert


async def test_analyst_without_tool_client_is_unchanged():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await make_analyst(provider)(_state())
    assert out["assessment"].rationale == "Base."


async def test_analyst_survives_tool_failure():
    provider = FakeProvider(responses={CandidateAssessment: CANNED})
    out = await make_analyst(provider, tool_client=_BoomClient())(_state())
    assert out["assessment"].candidate_name == "Ada Lovelace"  # still produced
    assert out["assessment"].rationale == "Base."  # no note appended on failure
```

`tests/api/test_mcp_integration.py`:

```python
import httpx
from httpx import ASGITransport

from employee_agent.api.app import create_app
from employee_agent.config import Settings
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import CandidateAssessment, VerifierVerdict

CANNED = CandidateAssessment(
    candidate_name="Ada Lovelace", years_experience=5.0, top_skills=["python"],
    skill_matches=[], overall_match_score=88, recommendation="advance", rationale="Base.",
)
ACCEPT = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")


async def test_assessment_includes_mcp_verification(tmp_path):
    s = Settings(_env_file=None, provider="fake", api_keys="key-a",
                 sqlite_path=str(tmp_path / "ck.sqlite"), chroma_path=str(tmp_path / "chroma"))
    app = create_app(s, provider=FakeProvider(responses={CandidateAssessment: CANNED, VerifierVerdict: ACCEPT}))
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        job_id = (await c.post("/jobs", headers={"x-api-key": "key-a"},
                               data={"job_description": "Senior Python"},
                               files={"resume": ("r.txt", b"Ada Lovelace. Python.", "text/plain")})).json()["job_id"]
        r = await c.get(f"/jobs/{job_id}", headers={"x-api-key": "key-a"})
        assert "verify_certification" in r.json()["assessment"]["rationale"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/engine/test_analyst_tool.py tests/api/test_mcp_integration.py -v`
Expected: FAIL (`make_analyst` has no `tool_client`; API rationale has no note).

- [ ] **Step 3: Extend `make_analyst` in `employee_agent/engine/nodes.py`**

Change the signature and add the tool call. Replace the `make_analyst` factory with:

```python
def make_analyst(provider: Provider, tool_client=None):
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
        if tool_client is not None and "verify_certification" in getattr(role, "tool_allowlist", []):
            cert = assessment.top_skills[0] if assessment.top_skills else "n/a"
            try:
                result = await tool_client.call(
                    "verify_certification",
                    {"name": assessment.candidate_name, "certification": cert},
                )
                note = f" [MCP verify_certification: {cert} -> verified={result.get('verified')}]"
                assessment = assessment.model_copy(
                    update={"rationale": assessment.rationale + note}
                )
            except Exception:  # noqa: BLE001 - resilient: continue without the tool
                pass
        return {"assessment": assessment}

    return analyst
```

- [ ] **Step 4: Extend `build_graph` in `employee_agent/engine/graph.py`**

Change the signature and the analyst wiring:

```python
def build_graph(provider: Provider, retriever: Retriever, checkpointer=None, tool_client=None):
```

and

```python
    g.add_node("analyst", make_analyst(provider, tool_client=tool_client))
```

- [ ] **Step 5: Wire the client into `employee_agent/api/app.py`**

Add the import:

```python
from employee_agent.mcp_tools.client import MCPToolClient
```

Add a graph helper (next to the other helpers, above `create_app`):

```python
def _graph_for(app, saver):
    return build_graph(
        app.state.provider, app.state.retriever,
        checkpointer=saver, tool_client=app.state.tool_client,
    )
```

In `create_app`, set the client on state (after `app.state.retriever = ...`):

```python
    app.state.tool_client = MCPToolClient(allowlist={"verify_certification"})
```

Then replace the three in-route `build_graph(request.app.state.provider, request.app.state.retriever, checkpointer=saver)` calls (in `create_job`, `get_job`, `approve_job`, `job_trace`) with:

```python
        graph = _graph_for(request.app, saver)
```

- [ ] **Step 6: Run the affected tests**

Run: `.venv/bin/python -m pytest tests/engine/test_analyst_tool.py tests/api/test_mcp_integration.py tests/engine/test_nodes.py -v`
Expected: PASS (3 + 1 + 4 = 8 passed; the existing analyst test still green because `tool_client` defaults to `None`).

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all prior + Plan 6; ~85 passed).

- [ ] **Step 8: Commit**

```bash
git add employee_agent/engine/nodes.py employee_agent/engine/graph.py employee_agent/api/app.py tests/engine/test_analyst_tool.py tests/api/test_mcp_integration.py
git commit -m "feat: analyst calls MCP verify_certification tool (allowlisted, resilient)"
```

---

## Definition of Done (Plan 6)

- `.venv/bin/python -m pytest -q` is green across the MCP server/client tests, the analyst-tool tests, and all prior modules.
- The MCP server exposes `verify_certification` and the client calls it over the real MCP protocol **in-memory** (no subprocess, offline).
- When the active role allowlists the tool, the Analyst calls it and the verification note appears in the assessment `rationale` — visible end-to-end through `GET /jobs/{id}`.
- A failing tool call is swallowed: the job still completes (resilience).
- Backward compatible: `make_analyst(provider)` / `build_graph(...)` without a `tool_client` behave exactly as before.

## Next Plan

Plan 7 — **Streamlit demo UI**: upload résumé + JD, start a job, stream the node progress, show the HITL approval screen (approve/edit/reject), and render the final `CandidateAssessment` — talking to the FastAPI endpoints from Plan 5. It depends on the running API.
