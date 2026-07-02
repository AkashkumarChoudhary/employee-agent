# Employee Agent — Plan 2: RAG Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the RAG knowledge layer — document ingestion (LangChain loaders/splitters), a Chroma vector store namespaced per job, and a retriever that embeds through the Plan 1 `Provider` abstraction — so the graph (Plan 3) can index resume/JD text and retrieve the chunks relevant to each requirement.

**Architecture:** Three small, single-responsibility modules under `employee_agent/rag/`. `ingest.py` turns raw files (PDF/text) into clean text and `Chunk` lists via LangChain's `RecursiveCharacterTextSplitter`. `store.py` wraps a local Chroma client (ephemeral in tests, persistent in prod) and always accepts **precomputed** embeddings, so Chroma never runs its own embedding model. `retriever.py` composes a `Provider` (for embeddings) with a `VectorStore` to give `index()` / `retrieve()`. All embeddings in tests come from `FakeProvider` (deterministic, offline); Chroma telemetry is disabled so nothing touches the network.

**Tech Stack:** Python 3.11+, Pydantic v2, `langchain-text-splitters`, `pypdf`, `chromadb`, pytest, pytest-asyncio.

## Global Constraints

- Python **3.11+** (uses `X | None`, `Literal`, `TypedDict`).
- Pydantic **v2**.
- **Tests never hit the network and never cost money** — embeddings come from `FakeProvider`; Chroma runs locally with `anonymized_telemetry=False`; no embedding model is ever downloaded (we always pass precomputed embeddings).
- Reuse Plan 1 interfaces verbatim: `employee_agent.schemas.Chunk(text: str, source: str, score: float)`, `employee_agent.providers.base.Provider` (`async embed(texts: list[str]) -> list[list[float]]`), `employee_agent.providers.fake.FakeProvider`.
- Package root: `employee_agent/`. Tests root: `tests/`. Run tests with `.venv/bin/python -m pytest`.
- All work happens on the current feature branch (`feat/employee-agent-foundations`, which carries Plan 1's code); commit per task.

---

### Task 1: RAG dependencies & document ingestion (loaders/splitters)

**Files:**
- Modify: `pyproject.toml` (add `langchain-text-splitters`, `pypdf`, `chromadb` to `dependencies`)
- Create: `employee_agent/rag/__init__.py`
- Create: `employee_agent/rag/ingest.py`
- Test: `tests/rag/test_ingest.py`

**Interfaces:**
- Consumes: `employee_agent.schemas.Chunk`.
- Produces (importable from `employee_agent.rag.ingest`):
  - `load_pdf(path: str) -> str` — concatenates text from every PDF page.
  - `load_document(path: str) -> str` — dispatches by suffix: `.pdf` → `load_pdf`, otherwise reads the file as UTF-8 text.
  - `split_text(text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Chunk]` — splits with `RecursiveCharacterTextSplitter`; every returned `Chunk` has the given `source` and `score=0.0` (retrieval sets the real score later).

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

Replace the `dependencies` array so it reads:

```toml
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "google-genai>=0.3",
    "ollama>=0.3",
    "langchain-text-splitters>=0.2",
    "pypdf>=4.0",
    "chromadb>=0.5",
]
```

- [ ] **Step 2: Install the new dependencies**

Run: `.venv/bin/python -m pip install "langchain-text-splitters>=0.2" "pypdf>=4.0" "chromadb>=0.5"`
Expected: installs successfully (chromadb pulls onnxruntime etc.; this is fine, we never invoke its embedder).

- [ ] **Step 3: Create `employee_agent/rag/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Write the failing test** in `tests/rag/test_ingest.py`

```python
from employee_agent.rag import ingest
from employee_agent.schemas import Chunk


def test_split_text_produces_chunks_with_source_and_zero_score():
    text = "Python developer. " * 200  # long enough to require splitting
    chunks = ingest.split_text(text, source="resume", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "resume" for c in chunks)
    assert all(c.score == 0.0 for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_split_text_short_text_single_chunk():
    chunks = ingest.split_text("short", source="job_description")
    assert len(chunks) == 1
    assert chunks[0].text == "short"
    assert chunks[0].source == "job_description"


def test_load_document_reads_text_file(tmp_path):
    p = tmp_path / "resume.txt"
    p.write_text("Ada Lovelace\nPython, Math", encoding="utf-8")
    assert ingest.load_document(str(p)) == "Ada Lovelace\nPython, Math"


def test_load_document_dispatches_pdf(monkeypatch, tmp_path):
    called = {}

    def fake_load_pdf(path):
        called["path"] = path
        return "PDF TEXT"

    monkeypatch.setattr(ingest, "load_pdf", fake_load_pdf)
    p = tmp_path / "resume.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    assert ingest.load_document(str(p)) == "PDF TEXT"
    assert called["path"] == str(p)
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rag/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.rag.ingest'`

- [ ] **Step 6: Implement `employee_agent/rag/ingest.py`**

```python
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from employee_agent.schemas import Chunk


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: str) -> str:
    if Path(path).suffix.lower() == ".pdf":
        return load_pdf(path)
    return Path(path).read_text(encoding="utf-8")


def split_text(
    text: str, source: str, chunk_size: int = 800, chunk_overlap: int = 100
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return [
        Chunk(text=piece, source=source, score=0.0)
        for piece in splitter.split_text(text)
    ]
```

> `load_document` calls the module-global `load_pdf`, so `monkeypatch.setattr(ingest, "load_pdf", ...)` in the test intercepts it.

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rag/test_ingest.py -v`
Expected: PASS (4 passed)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml employee_agent/rag/__init__.py employee_agent/rag/ingest.py tests/rag/test_ingest.py
git commit -m "feat: rag document ingestion (loaders + splitters)"
```

---

### Task 2: Chroma vector store wrapper

**Files:**
- Create: `employee_agent/rag/store.py`
- Test: `tests/rag/test_store.py`

**Interfaces:**
- Consumes: `employee_agent.schemas.Chunk`, `chromadb`.
- Produces: `employee_agent.rag.store.VectorStore(path: str | None = None)` — ephemeral (in-memory) when `path` is `None`, else a `PersistentClient` at `path`. Telemetry disabled.
  - `add(namespace: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None` — upserts into the per-namespace collection; no-op when `chunks` is empty. Stores `chunk.source` in metadata.
  - `query(namespace: str, query_embedding: list[float], k: int = 4) -> list[Chunk]` — returns up to `k` `Chunk`s nearest to `query_embedding`, ordered nearest-first, each with `score = 1.0 / (1.0 + distance)` (so identical vectors → score `1.0`). Returns `[]` for an empty collection.

- [ ] **Step 1: Write the failing test** in `tests/rag/test_store.py`

```python
from employee_agent.rag.store import VectorStore
from employee_agent.schemas import Chunk


def _emb(seed: float, dim: int = 8):
    return [seed] * dim


def test_add_and_query_returns_nearest_first():
    store = VectorStore()  # ephemeral, in-memory
    chunks = [
        Chunk(text="python expert", source="resume", score=0.0),
        Chunk(text="java developer", source="resume", score=0.0),
    ]
    store.add("job-1", chunks, [_emb(0.1), _emb(0.9)])
    results = store.query("job-1", _emb(0.1), k=2)  # identical to first chunk
    assert len(results) == 2
    assert results[0].text == "python expert"
    assert results[0].source == "resume"
    assert results[0].score >= results[1].score
    assert results[0].score == 1.0  # exact-match distance 0 -> score 1.0


def test_query_respects_k():
    store = VectorStore()
    chunks = [Chunk(text=f"c{i}", source="resume", score=0.0) for i in range(5)]
    store.add("job-2", chunks, [[float(i)] * 8 for i in range(5)])
    results = store.query("job-2", [0.0] * 8, k=3)
    assert len(results) == 3


def test_namespaces_are_isolated():
    store = VectorStore()
    store.add("ns-a", [Chunk(text="alpha", source="resume", score=0.0)], [[0.1] * 8])
    store.add("ns-b", [Chunk(text="beta", source="resume", score=0.0)], [[0.1] * 8])
    a = store.query("ns-a", [0.1] * 8, k=5)
    assert [c.text for c in a] == ["alpha"]


def test_add_empty_is_noop():
    store = VectorStore()
    store.add("ns-empty", [], [])
    assert store.query("ns-empty", [0.1] * 8, k=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rag/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.rag.store'`

- [ ] **Step 3: Implement `employee_agent/rag/store.py`**

```python
import chromadb
from chromadb.config import Settings

from employee_agent.schemas import Chunk

_SETTINGS = Settings(anonymized_telemetry=False)


class VectorStore:
    def __init__(self, path: str | None = None):
        self._client = (
            chromadb.PersistentClient(path=path, settings=_SETTINGS)
            if path
            else chromadb.EphemeralClient(settings=_SETTINGS)
        )

    def _collection(self, namespace: str):
        # embedding_function=None: we always supply precomputed embeddings,
        # so Chroma never loads its default (ONNX) embedder or hits the network.
        return self._client.get_or_create_collection(
            name=namespace, embedding_function=None
        )

    def add(
        self, namespace: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        if not chunks:
            return
        col = self._collection(namespace)
        col.upsert(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def query(
        self, namespace: str, query_embedding: list[float], k: int = 4
    ) -> list[Chunk]:
        col = self._collection(namespace)
        res = col.query(query_embeddings=[query_embedding], n_results=k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            Chunk(text=doc, source=(meta or {}).get("source", ""),
                  score=1.0 / (1.0 + dist))
            for doc, meta, dist in zip(docs, metas, dists)
        ]
```

> **Watch-point (resolve during green):** if the installed `chromadb` rejects `embedding_function=None`, define a trivial `EmbeddingFunction` subclass whose `__call__` raises (it is never invoked because we always pass `embeddings=`) and pass an instance instead. Do **not** fall back to the default embedder — that would download an ONNX model.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rag/test_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add employee_agent/rag/store.py tests/rag/test_store.py
git commit -m "feat: chroma vector store wrapper (namespaced, byo-embeddings)"
```

---

### Task 3: Retriever (embed → index → retrieve)

**Files:**
- Create: `employee_agent/rag/retriever.py`
- Test: `tests/rag/test_retriever.py`

**Interfaces:**
- Consumes: `employee_agent.providers.base.Provider` (`embed`), `employee_agent.rag.store.VectorStore`, `employee_agent.schemas.Chunk`. Tested with `employee_agent.providers.fake.FakeProvider`.
- Produces: `employee_agent.rag.retriever.Retriever(provider: Provider, store: VectorStore)`
  - `async index(namespace: str, chunks: list[Chunk]) -> None` — embeds all chunk texts via the provider and adds them to the store under `namespace`; no-op for empty `chunks`.
  - `async retrieve(namespace: str, query: str, k: int = 4) -> list[Chunk]` — embeds `query`, returns up to `k` nearest chunks (nearest-first, with scores). `[]` when the namespace is empty.

- [ ] **Step 1: Write the failing test** in `tests/rag/test_retriever.py`

```python
from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.schemas import Chunk


def _chunks(texts, source="resume"):
    return [Chunk(text=t, source=source, score=0.0) for t in texts]


async def test_index_then_retrieve_exact_match_ranks_first():
    retr = Retriever(FakeProvider(embed_dim=16), VectorStore())
    await retr.index(
        "job-1", _chunks(["python and django", "rust systems", "graphic design"])
    )
    results = await retr.retrieve("job-1", "python and django", k=2)
    # identical text -> identical FakeProvider embedding -> distance 0 -> ranks first
    assert results[0].text == "python and django"
    assert results[0].score == 1.0


async def test_retrieve_returns_at_most_k():
    retr = Retriever(FakeProvider(embed_dim=16), VectorStore())
    await retr.index("job-2", _chunks([f"skill {i}" for i in range(6)]))
    results = await retr.retrieve("job-2", "skill 0", k=3)
    assert len(results) == 3
    assert all(isinstance(c, Chunk) for c in results)


async def test_index_empty_then_retrieve_is_empty():
    retr = Retriever(FakeProvider(), VectorStore())
    await retr.index("job-empty", [])
    assert await retr.retrieve("job-empty", "anything", k=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rag/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'employee_agent.rag.retriever'`

- [ ] **Step 3: Implement `employee_agent/rag/retriever.py`**

```python
from employee_agent.providers.base import Provider
from employee_agent.rag.store import VectorStore
from employee_agent.schemas import Chunk


class Retriever:
    def __init__(self, provider: Provider, store: VectorStore):
        self._provider = provider
        self._store = store

    async def index(self, namespace: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = await self._provider.embed([c.text for c in chunks])
        self._store.add(namespace, chunks, embeddings)

    async def retrieve(self, namespace: str, query: str, k: int = 4) -> list[Chunk]:
        embeddings = await self._provider.embed([query])
        return self._store.query(namespace, embeddings[0], k=k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/rag/test_retriever.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all Plan 1 + Plan 2 tests green; 29 + 11 = 40 passed)

- [ ] **Step 6: Commit**

```bash
git add employee_agent/rag/retriever.py tests/rag/test_retriever.py
git commit -m "feat: rag retriever (embed + index + retrieve)"
```

---

## Definition of Done (Plan 2)

- `.venv/bin/python -m pytest -q` is green across ingest, store, retriever, and all Plan 1 modules.
- `Retriever(FakeProvider(), VectorStore()).index()` then `.retrieve()` returns nearest-first chunks with scores; an exact-text query ranks its chunk first with `score == 1.0`.
- Vector store namespaces are isolated; empty adds/queries are safe no-ops.
- No network calls and no model downloads in the test suite (Chroma telemetry disabled; embeddings always precomputed via `FakeProvider`).

## Implementation Notes (discovered during execution)

- **chromadb 1.5.9** was installed (the `>=0.5` floor resolved to 1.x). The watch-point held up: `get_or_create_collection(name=..., embedding_function=None)` works and never downloads a model; an exact-vector query returns distance `0.0` (→ score `1.0`).
- **Ephemeral chromadb is one instance per process.** `chromadb.EphemeralClient()` shares a single in-memory backing store across all `VectorStore()` instances (and rejects a second ephemeral instance with different settings). Per-instance databases can't be cleanly auto-created. Consequence: tests reusing the same namespace with different embedding dimensions collide (`InvalidArgumentError: Collection expecting embedding with dimension of N`). Resolved by giving each test file distinct namespace prefixes (`store-*`, `retr-*`); production is unaffected because it uses a persistent path namespaced per unique `job_id`. This constraint is documented in `VectorStore`'s docstring.

## Next Plan

Plan 3 — **The graph**: LangGraph orchestrator-worker nodes (Manager → Parser → Retriever → Analyst) over the `AgentState` `TypedDict`, a SQLite checkpointer, and Gemini structured output. It will depend on `ingest.split_text`/`load_document`, `Retriever`, `build_provider`, and the Plan 1 schemas.
