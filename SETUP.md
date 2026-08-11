# Setup on a new machine

Everything below was run end-to-end on a clean venv and a clean Docker build.

## 1. What you need to bring

| | Why |
|---|---|
| **The repo** (`git clone`) | Nothing machine-specific is stored outside git. |
| **A Gemini API key** | The only secret. Free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Without it the agent runs but returns an **empty** assessment (see §5). |
| **Internet** | For `pip install` and for Gemini API calls at demo time. |

`.venv/` and `data/` are gitignored and regenerated locally — do **not** copy them between machines.

## 2. Prerequisites

- **Python 3.11+** (verified on 3.12) and the venv module:
  `sudo apt install python3 python3-venv` on Debian/Ubuntu; `brew install python@3.12` on macOS.
- **Docker 24+ with Compose v2** — only if you take the Docker path (verified on Docker 29.6 / Compose v5.3).
- Ollama is **not** required. It's only used as a failover provider, and `ENABLE_FAILOVER` defaults to false in `.env.example`.

## 3. Path A — local Python (fastest to iterate)

```bash
git clone <repo-url> employee-agent && cd employee-agent

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.lock.txt     # exact, tested versions
pip install -e ".[dev]"                  # the package itself + pytest

pytest -q                                # expect: 92 passed — offline, no API key needed
```

Configure and run:

```bash
cp .env.example .env
# edit .env: set GEMINI_API_KEY, keep PROVIDER=gemini and ENABLE_FAILOVER=false

# terminal 1 — API on :8000
uvicorn employee_agent.api.app:create_app --factory --reload

# terminal 2 — UI on :8501
EMPLOYEE_AGENT_API_URL=http://localhost:8000 \
EMPLOYEE_AGENT_API_KEY=demo-key \
  streamlit run employee_agent/ui/streamlit_app.py
```

Open <http://localhost:8501>. `EMPLOYEE_AGENT_API_KEY` must match a key in `API_KEYS` in `.env`.

## 4. Path B — Docker (what the demo script assumes)

```bash
cp .env.example .env      # compose reads .env automatically; set GEMINI_API_KEY here
docker compose up --build # first build ~2-4 min; API :8000, UI :8501
```

If `GEMINI_API_KEY` is unset **and** `PROVIDER=gemini`, the API container exits at startup with
`ValueError: gemini_api_key is required when provider='gemini'`. Set the key, or set `PROVIDER=fake`.

State (SQLite checkpoints + Chroma index) lives in the `agent-data` volume. `docker compose down -v` resets it.

## 5. Verify it works

```bash
curl localhost:8000/health                                  # {"status":"ok"}
curl localhost:8000/whoami -H "x-api-key: demo-key"         # echoes the key
curl -o /dev/null -w '%{http_code}\n' localhost:8000/whoami # 401 — auth is on
```

Then in the UI: paste a job description, upload a `.pdf`/`.txt`/`.md` résumé, **Run assessment** →
the run pauses at `awaiting_human` → Approve/Edit/Reject → open **Execution trace** to show the
node path including the verifier's retry loop.

**Offline smoke test** (no key, no network — proves the plumbing only):

```bash
PROVIDER=fake API_KEYS=demo-key uvicorn employee_agent.api.app:create_app --factory
```

`FakeProvider` fills the schema with type defaults, so the assessment renders blank
(`candidate_name: ""`, `overall_match_score: 0`, `recommendation: null`). Fine for checking the
graph, auth, HITL and MCP wiring; **not** a presentable demo. Use a real key for that.

## 6. Ports and troubleshooting

| Symptom | Cause / fix |
|---|---|
| `8000`/`8501` already in use | `uvicorn --port 8010`, `streamlit run … --server.port 8511`, and point `EMPLOYEE_AGENT_API_URL` at the new API port. |
| `ImportError: cannot import name 'create_connected_server_and_client_session'` | You installed `mcp` 2.x. Dependencies are pinned to `mcp>=1.0,<2` — install via `requirements.lock.txt`, not by hand. |
| UI shows a 401 / connection error | `EMPLOYEE_AGENT_API_KEY` isn't in the API's `API_KEYS`, or the API isn't up yet. |
| Job created before an API restart 404s | `JobStore` is in-memory by design; ownership records don't survive a restart. Create a fresh job. |
| Gemini quota / 429 | Free-tier rate limits. Wait, or switch `PROVIDER=fake` to rehearse the flow. |
| `pip install` compiling from source | Use Python 3.11 or 3.12 — the locked versions ship wheels for those. |

## 7. Demo-day checklist

1. `docker compose up` (or both local processes) — confirm `/health`.
2. Have a résumé file and a job description ready in a scratch folder.
3. Run one throwaway assessment before presenting — it warms the Chroma collection and confirms the key works.
4. Two beats from the original demo script are **not** built: live node streaming in the UI, and
   swapping to a second role preset (only `hr_analyst` is registered). Use the **Execution trace**
   expander to show the graph path instead.
