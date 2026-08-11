# Employee Agent

A configurable **"digital employee" engine** — an HR/recruiting analyst as the hero use case.

It's a LangGraph orchestrator-worker graph — **RAG → analysis → CRAG/Self-RAG verification → bounded retry → human-in-the-loop → finalize** — served over a FastAPI REST API, demoed in Streamlit, calling a real **MCP** tool. Swap a `RoleConfig` (extraction schema + prompts + knowledge namespace + tool allowlist) and the same engine becomes a different "employee," so it covers HR, research-analyst, knowledge-assistant, and support use cases without four separate apps.

Built phase-by-phase (spec + plans in [`docs/superpowers/`](docs/superpowers/)), test-first, **90+ tests, fully offline and deterministic**.

## Architecture

```text
[Streamlit UI] --HTTP--> [FastAPI: X-API-Key auth, BOLA ownership, rate-limit, upload validation]
                               |
                               v
   manager -> parser -> retriever (RAG / Chroma) -> analyst --(MCP verify_certification, allowlisted)
                                                        |
                                                        v
                             verifier (CRAG / Self-RAG) --accept--> gate -> hitl (interrupt) -> finalizer
                                   |  ^  bounded retry (<= MAX_RETRIES)
                                   v  |
                             retriever / analyst
   (conversation state persisted + resumable via the SQLite checkpointer — "time travel")
```

- **Manager** loads the active `RoleConfig` and owns the shared `AgentState`.
- **Parser** cleans + splits the résumé (LangChain splitters + pypdf) and indexes it.
- **Retriever** embeds and fetches the chunks relevant to the job description (Chroma, namespaced per job).
- **Analyst** produces a Pydantic-validated `CandidateAssessment` (structured output) and, when the role allows, calls the **MCP** `verify_certification` tool.
- **Verifier** grades whether the claims are grounded (CRAG/Self-RAG); if not, it loops back — **bounded** so it always terminates.
- **HITL gate** pauses with `interrupt()` for a human to approve / edit / reject; **Finalizer** commits the result.

## Tech-stack tiers

🟢 fully built · 🟡 real but minimal · 🔵 config/docs (deploy-ready, **not** operated)

| Technology | Where it lives | Tier |
|---|---|---|
| Python 3.11+ | Whole codebase | 🟢 |
| Pydantic v2 | Contracts, `AgentState`, structured-output enforcement | 🟢 |
| LangChain (splitters) + pypdf | Document ingestion | 🟢 |
| Chroma | RAG vector store, namespaced per job | 🟢 |
| Gemini (LLM + embeddings) | Primary provider via `providers/` | 🟢 |
| LangGraph | Orchestrator-worker graph, conditional routing, retry loop | 🟢 |
| SQLite | LangGraph checkpointer (durable/resumable state) | 🟢 |
| Human-in-the-Loop | `interrupt()` gate + `/approve` endpoint + UI screen | 🟢 |
| CRAG / Self-RAG | Verifier node: grading + bounded retry | 🟢 |
| FastAPI | REST API wrapping the graph | 🟢 |
| Streamlit | Demo UI: upload → assess → approve → trace | 🟢 |
| API security | X-API-Key auth, slowapi rate limiting, Pydantic validation, BOLA ownership | 🟢 |
| MCP | Mock MCP server + in-memory client, wired into the Analyst (allowlisted) | 🟡 |
| Ollama | Configured failover provider | 🟡 |
| Light eval harness | `employee_agent/eval/` — labeled cases, accuracy metric | 🟡 |
| Docker / Compose | `docker compose up` (API + UI) | 🟢 |
| Cloud Run / Vertex AI | `deploy/cloud-run.md` — documented deploy target | 🔵 |
| GKE / Apigee / Cloud Armor | `deploy/cloud-run.md` — documented scale path | 🔵 |
| GitHub Actions CI | `.github/workflows/ci.yml` — lint/test | 🔵 |

**Honesty rule:** 🔵 items are config + docs, not running infrastructure.

## Quickstart

Full machine setup — prerequisites, both run paths, troubleshooting — is in [`SETUP.md`](SETUP.md).

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock.txt    # exact, tested versions
pip install -e ".[dev]"

pytest -q                       # 92 tests, fully offline (no network, no cost)

# Run the API + UI locally
cp .env.example .env            # set GEMINI_API_KEY; PROVIDER=fake works offline
uvicorn employee_agent.api.app:create_app --factory --reload
EMPLOYEE_AGENT_API_URL=http://localhost:8000 EMPLOYEE_AGENT_API_KEY=demo-key \
  streamlit run employee_agent/ui/streamlit_app.py

# Or with Docker
docker compose up --build       # API on :8000, UI on :8501
```

> `PROVIDER=fake` exercises the whole graph offline but fills the assessment with schema
> defaults (blank name, score 0) — use a real `GEMINI_API_KEY` for a presentable demo.

## API

Every endpoint except `/health` requires an `X-API-Key` header. Request/response bodies are Pydantic models.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/jobs` | Create a job (résumé upload + JD); runs the graph to the human gate; returns `job_id` |
| `GET` | `/jobs/{id}` | Fetch status + draft/final assessment (ownership-checked) |
| `POST` | `/jobs/{id}/approve` | Submit the human decision (`approve` / `edit` / `reject`); resumes the graph |
| `GET` | `/jobs/{id}/trace` | Node-by-node execution path |
| `GET` | `/health` | Liveness (open) |
| `GET` | `/whoami` | Echoes the caller's key (authed) |

## Security

- **AuthN** — `X-API-Key` on every non-health endpoint (keys in env; documented upgrade: OAuth2/JWT).
- **AuthZ / BOLA** — each job is owned by its creating key; cross-owner access returns 404.
- **Resource consumption** — slowapi per-key rate limiting + a hard graph-iteration cap (`MAX_RETRIES`) that prevents runaway agent loops.
- **Input validation** — Pydantic on all inputs; upload type/size limits; per-role **MCP tool allowlist** (mitigates SSRF-via-MCP).

## Testing

`pytest -q` is **fully offline and deterministic** — provider SDKs are mocked, `FakeProvider` is used elsewhere, `temperature=0`, and no embedding model is ever downloaded. A light eval harness (`employee_agent/eval/`) runs labeled candidates through the real graph and reports recommendation accuracy ("metrics, not vibe-checks").

## Deploy & scale path

See [`deploy/cloud-run.md`](deploy/cloud-run.md). Cloud Run, Vertex AI, GKE, Apigee, and Cloud Armor are **designed-for, not running** — the provider abstraction swaps Gemini → Vertex AI with no app changes.

## Status & future work

Plans 1–8 are complete. Future work: flesh out the research/knowledge/support role presets beyond stubs, promote the Cloud Run config from 🔵 to a running deployment, and add live node-streaming to the UI.
