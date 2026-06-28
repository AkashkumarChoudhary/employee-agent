# Employee Agent — Design Spec

**Date:** 2026-06-19
**Status:** Approved (brainstorming complete; ready for implementation planning)
**Source context:** Derived from `genai-project-notebooklm-context.md` (NotebookLM "GenAI Project" chat), Turn 8 onward.

---

## 1. Summary

A domain-agnostic **"digital employee" engine**: a LangGraph orchestrator-worker state
machine, exposed over a FastAPI REST API, demoed as an **HR/recruiting analyst** through a
Streamlit UI. The same engine becomes a different "employee" by swapping a `RoleConfig`
(extraction schema + prompts + knowledge namespace + tool allowlist), so it covers the HR,
research-analyst, knowledge-assistant, and support-agent use cases without building four
separate apps.

The project is a **portfolio capstone**. Success = a clear, demo-able, explainable system
that shows breadth across GenAI + backend + deployment, runs reproducibly via
`docker compose up`, and is honest about what is built vs. documented.

### Goals
- Demonstrate breadth across the full GenAI lifecycle (RAG → multi-agent orchestration →
  structured extraction → HITL → API → deploy) in one coherent project.
- Be reliably demo-able in a ~3-minute interview walkthrough.
- Use **every technology named** in the Turn 8–12 blueprint, each with a concrete home.

### Non-goals (explicitly out of scope)
- Training/fine-tuning foundational models (use Gemini API; Ollama as local failover).
- A production React/Next.js frontend (Streamlit demo UI only).
- Multimodal (audio/video) processing — text + PDF only.
- A live, running GKE/Apigee production deployment (designed and documented, not operated).

### Constraints
- Effort budget: ~1–2 weeks focused.
- Cost: free-tier first — **Google Gemini** (LLM + embeddings, AI Studio free tier),
  local Chroma, local SQLite, local Ollama failover.

---

## 2. Architecture Overview

One sentence: a LangGraph orchestrator-worker graph wrapped by FastAPI, with persistent
state in SQLite, a RAG knowledge layer in Chroma, an MCP tool layer, and a Streamlit demo UI.

```
                 ┌──────────── SQLite checkpointer (state + "time travel") ─────────────┐
                 │                                                                       │
 [API/UI] → Manager(router) ─► Intake/Parser ─► Retriever (RAG) ─► Analyst/Matcher ─► Verifier (CRAG/Self-RAG)
                 ▲                                     │ retrieve          │ MCP tools      │
                 │                                     ▼                   ▼                ▼
                 └──────── Finalizer ◄──── HITL gate (interrupt) ◄──────────── (pass / bounded retry loop)
```

### Module boundaries
Each module has one clear purpose, a defined interface, and is independently testable.

| Module | Purpose | Key dependencies |
|--------|---------|------------------|
| `engine/` | LangGraph graph definition, nodes, `AgentState` | langgraph, providers, rag, mcp |
| `roles/` | `RoleConfig` presets (HR built; others stubbed) | pydantic |
| `rag/` | Document ingestion + retrieval | langchain, chroma, providers |
| `mcp/` | MCP client wiring + one mock MCP server | mcp sdk |
| `providers/` | LLM + embeddings abstraction; Gemini primary, Ollama failover | gemini sdk, ollama |
| `api/` | FastAPI app, auth, rate limiting, ownership checks | fastapi, slowapi |
| `ui/` | Streamlit demo UI | streamlit |
| `tests/` | Unit, integration, API tests + light eval harness | pytest |

---

## 3. Components

### LangGraph nodes (workers)
- **Manager / Router** — entry node; loads the active `RoleConfig` (HR preset), owns the
  shared `AgentState`, decides the path.
- **Intake/Parser** — LangChain loaders + splitters; resume (PDF) + JD → clean text + chunks.
- **Retriever (RAG)** — embeds chunks (Gemini embeddings) into Chroma (namespaced per job);
  retrieves chunks relevant to each JD requirement.
- **Analyst/Matcher** — reasoning core; produces the Pydantic-structured `CandidateAssessment`
  via structured output. May call MCP tools (allowlisted).
- **Verifier (CRAG/Self-RAG)** — grades whether retrieved context supports each extracted
  claim; on failure loops back to Retriever/Analyst (bounded, max 2 retries).
- **HITL gate** — LangGraph `interrupt()`; pauses for human approve/edit/reject before finalize.
- **Finalizer** — persists the approved assessment, returns structured JSON.

### Cross-cutting components
- **FastAPI** — REST endpoints wrapping the graph; async job execution; Pydantic contracts.
- **MCP layer** — LangGraph as MCP **client**; one mock MCP **server** (candidate-verification
  tool) so the integration is real.
- **Provider abstraction** — Gemini primary; Ollama failover on error/429.
- **LangSmith** — tracing across nodes (env-var toggle; no-ops cleanly when absent).
- **Streamlit UI** — upload, live step streaming, HITL approval screen, report view.
- **Docker Compose** — app + Chroma; `docker compose up`. Cloud Run config + GKE/Apigee notes
  as the documented scale path.

---

## 4. Data Flow (HR hero path)

1. **Upload** — Streamlit submits resume (PDF) + JD → `POST /jobs`. FastAPI validates
   (Pydantic), creates `job_id`, starts the graph async, returns `job_id`.
2. **Intake/Parser** — loads + splits both docs; writes `parsed_resume`, `job_description`.
3. **Retriever** — embeds resume chunks → Chroma (namespaced per `job_id`); retrieves chunks
   relevant to each JD requirement.
4. **Analyst/Matcher** — Gemini prompted with JD requirements + retrieved chunks, forced into
   the `CandidateAssessment` schema. Optionally calls an MCP tool.
5. **Verifier (CRAG/Self-RAG)** — grades grounding of each claim; unsupported/weak → bounded
   loop back to Retriever/Analyst; good → proceed.
6. **HITL gate** — graph `interrupt()`s; state checkpointed to SQLite. UI shows the draft;
   recruiter approves/edits/rejects. `POST /jobs/{id}/approve` resumes from the checkpoint.
7. **Finalizer** — persists the approved `CandidateAssessment`, marks `done`, returns JSON.

### Resilience hooks on this path
- Gemini 429/failure → retry w/ backoff → **Ollama** failover.
- Weak retrieval → CRAG re-query (bounded).
- Bad reasoning → Verifier loop (bounded) → escalate to HITL.
- Crash mid-job → resume from **SQLite checkpoint** ("time travel").
- Agent loop / no convergence → hard iteration cap → stop, surface to human.
- Bad MCP tool response → caught, agent continues without it, flagged in trace.

---

## 5. Data Model (Pydantic contracts)

```python
# Request
class CreateJobRequest(BaseModel):
    role: str = "hr_analyst"            # selects the role preset
    job_description: str
    # resume arrives as a multipart file upload

# Shared graph state (LangGraph AgentState)
class AgentState(TypedDict):
    job_id: str
    role_config: RoleConfig            # schema + prompts + tool allowlist
    job_description: str
    parsed_resume: str
    retrieved_chunks: list[Chunk]
    assessment: CandidateAssessment | None
    verifier_verdict: VerifierVerdict | None
    retry_count: int
    status: Literal["running", "awaiting_human", "done", "error"]

# A retrieved RAG chunk
class Chunk(BaseModel):
    text: str
    source: str                         # e.g. "resume" | "job_description"
    score: float                        # retrieval similarity

# Structured extraction (the headline output)
class SkillMatch(BaseModel):
    requirement: str
    candidate_evidence: str | None      # cited from resume
    met: bool
    confidence: float

class CandidateAssessment(BaseModel):
    candidate_name: str
    years_experience: float
    top_skills: list[str]
    skill_matches: list[SkillMatch]
    overall_match_score: int            # 0-100
    recommendation: Literal["advance", "hold", "reject"]
    rationale: str
    human_approved: bool = False

class VerifierVerdict(BaseModel):
    grounded: bool
    unsupported_claims: list[str]
    action: Literal["accept", "retry_retrieval", "retry_analysis"]

# RoleConfig = how one "employee role" is defined (domain-agnostic engine)
class RoleConfig(BaseModel):
    name: str
    system_prompt: str
    extraction_schema: str              # which Pydantic model to enforce
    tool_allowlist: list[str]
    knowledge_namespace: str
```

**Persistence (SQLite):**
- (a) LangGraph **checkpointer** — graph state per `job_id` (pause/resume/time-travel).
- (b) **jobs table** — `job_id`, `role`, `owner_key`, `status`, final assessment JSON, timestamps.

The `RoleConfig` is the mechanism for "does all four jobs": swapping it retargets the same
engine. HR is fully built; research/knowledge/support ship as preset stubs.

---

## 6. Tech-Stack Mapping

Build tiers: 🟢 fully built · 🟡 real but minimal · 🔵 config/docs (deploy-ready, not exercised).

| # | Technology | Where it lives | Tier |
|---|------------|----------------|------|
| 1 | Python 3.11+ | Whole codebase | 🟢 |
| 2 | Pydantic | Contracts, `AgentState`, structured-output enforcement | 🟢 |
| 3 | LangChain | Loaders, splitters, embeddings wrapper, output parsers, Chroma integration | 🟢 |
| 4 | Chroma (vector store) | RAG index, namespaced per job/role | 🟢 |
| 5 | Gemini (LLM + embeddings) | Primary provider via `providers/`; AI Studio free tier | 🟢 |
| 6 | LangGraph | Orchestrator-worker graph, conditional routing, retry loops | 🟢 |
| 7 | SQLite | LangGraph checkpointer (time-travel) + jobs table | 🟢 |
| 8 | Human-in-the-Loop | `interrupt()` gate + `/approve` endpoint + UI screen | 🟢 |
| 9 | CRAG / Self-RAG | Verifier node: grading + bounded retry loop | 🟢 |
| 10 | FastAPI | REST API wrapping the graph; async jobs | 🟢 |
| 11 | Streamlit | Demo UI: upload, streaming, HITL approval, report | 🟢 |
| 12 | Docker / Docker Compose | App + Chroma containers; `docker compose up` | 🟢 |
| 13 | LangSmith | Tracing across nodes (env-var enabled) | 🟡 |
| 14 | MCP | LangGraph as client + one mock MCP server (verification tool) | 🟡 |
| 15 | Ollama | Configured failover provider | 🟡 |
| 16 | API security / OWASP Top 10 | API-key auth, slowapi rate limiting, Pydantic validation, BOLA ownership checks | 🟡 |
| 17 | Cloud Run | Dockerfile + service config + deploy script (documented real deploy target) | 🔵 |
| 18 | GKE / Apigee / Cloud Armor | "Scale path" doc: inference gateway, quota/Spike-Arrest, WAF | 🔵 |
| 19 | Vertex AI | Documented production swap-in for Gemini API (same provider abstraction) | 🔵 |
| 20 | CI/CD (GitHub Actions) | Lint + test workflow (stretch goal) | 🔵 |

**Honesty rule:** 🔵 items are config + docs, not running infrastructure. README and verbal
pitch label them "deploy-ready / designed-for," never "running in production."

---

## 7. Security, Observability & Error Handling

### Security (OWASP-aware, scoped to defensible)
- **AuthN** — `X-API-Key` header on every endpoint; keys in env. Documented upgrade: OAuth2/JWT.
- **AuthZ / BOLA** — each job owned by its creating key; `/jobs/{id}` checks ownership.
- **Rate limiting / resource consumption** — `slowapi` per-key limits + hard cap on graph
  iterations (prevents runaway agent loops). Documented gateway: Apigee Spike-Arrest + quotas.
- **Input validation** — Pydantic on all inputs; file-type/size limits on uploads; MCP tool
  **allowlist** per role (mitigates SSRF-via-MCP).
- **Secrets** — `.env` (gitignored) + `.env.example`; never logged.

### Observability
- **LangSmith** tracing across nodes (env toggle, no-ops when absent).
- **Structured JSON logging** with `job_id` correlation through every node.
- **`/health`** endpoint + **`GET /jobs/{id}/trace`** returning node-by-node execution path.

### Error handling
| Failure | Handling |
|---------|----------|
| Gemini 429 / API error | Retry w/ backoff → Ollama failover → mark `error` with reason |
| Weak/irrelevant retrieval | CRAG re-query (bounded) |
| Unsupported claims in output | Verifier → retry analysis (bounded) → escalate to HITL |
| Agent loop / no convergence | Hard iteration cap → stop, surface to human |
| Process crash mid-job | Resume from SQLite checkpoint (no lost work) |
| Bad MCP tool response | Caught; agent continues without it; flagged in trace |

No error path silently fails: it recovers, degrades gracefully, or escalates to the human.

---

## 8. API Surface (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/jobs` | Create a job (resume upload + JD); starts the graph; returns `job_id` |
| `GET` | `/jobs/{id}` | Fetch job status + draft/final assessment (ownership-checked) |
| `POST` | `/jobs/{id}/approve` | Submit human decision (approve/edit/reject); resumes the graph |
| `GET` | `/jobs/{id}/trace` | Node-by-node execution path for debugging/demo |
| `GET` | `/health` | Liveness check |

All endpoints require `X-API-Key`. Request/response bodies are Pydantic models.

---

## 9. Testing & Definition of Done

### Testing strategy
- **Unit** — each node in isolation with a mocked provider (`FakeProvider`): Parser, Retriever,
  Analyst (→ valid `CandidateAssessment`), Verifier (grounded vs. hallucinated → correct verdict).
  Pydantic schemas serve as a structural test layer.
- **Integration** — full graph on 2–3 fixture resume+JD pairs (stubbed provider); assert it
  reaches `awaiting_human`, resumes after approval, finalizes.
- **API** — FastAPI `TestClient`: auth required, BOLA blocked (key B can't read key A's job),
  rate limit triggers, validation rejects bad uploads.
- **Eval harness (light)** — ~5 labeled candidates with known advance/reject to sanity-check
  match scores ("metrics, not vibe-checks"). Stretch: log evals to LangSmith.
- **Determinism** — `temperature=0` + `FakeProvider` so tests never hit the network or cost money.

### Definition of Done
Runs via `docker compose up`; full HR flow works end-to-end through the UI including HITL;
tests green; README with architecture diagram + documented cloud/scale path. "Done" here means
**demo-able and reproducible**, with the production deploy documented (not a live GKE cluster).

### Demo script (≈3 minutes)
1. `docker compose up` → open Streamlit.
2. Upload resume + paste JD → watch nodes stream (Parser → Retriever → Analyst → Verifier).
3. Trigger the Verifier catching a weak/unsupported claim and looping — visible self-correction.
4. Hit the HITL approval screen → edit one field → approve.
5. Show the final structured `CandidateAssessment` JSON + the LangSmith trace.
6. Swap `role` to a stub preset to show the same engine becomes a different "employee."

---

## 10. Build Phases (1–2 week plan)

1. **Foundations** — repo skeleton, `providers/` (Gemini + Ollama failover), Pydantic schemas, config/env.
2. **RAG core** — ingestion (loaders/splitters), Chroma, retriever; tested on fixtures.
3. **The graph** — LangGraph nodes (Manager→Parser→Retriever→Analyst), SQLite checkpointer, structured output.
4. **Self-correction + HITL** — Verifier (CRAG/Self-RAG) retry loop, `interrupt()` gate.
5. **API** — FastAPI endpoints, auth, rate limiting, ownership checks, LangSmith.
6. **MCP** — mock MCP server + client wiring into the Analyst.
7. **UI** — Streamlit upload → stream → approve → report.
8. **Package & document** — Docker Compose, README + architecture diagram, Cloud Run config,
   GKE/Apigee scale-path notes, eval harness.

Roles 2–4 (research/knowledge/support) ship as **config stubs** demonstrating generality,
built only if time remains. CI/CD (GitHub Actions) is a stretch goal.

---

## 11. Open Questions / Future Work
- **Dynamic planning** — graft a ReAct-style planner onto the fixed graph as "future work."
- **Real cloud deploy** — promote Cloud Run config from 🔵 to running if the project continues
  past the portfolio milestone.
- **Additional role presets** — flesh out research/knowledge/support beyond stubs.
