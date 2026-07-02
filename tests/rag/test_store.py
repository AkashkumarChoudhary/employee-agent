from employee_agent.rag.store import VectorStore
from employee_agent.schemas import Chunk

# NOTE: chromadb allows only one ephemeral instance per process, so ephemeral
# VectorStore() instances share a backing store. Tests use distinct, prefixed
# namespaces so they never collide with each other or with test_retriever.py.


def _emb(seed: float, dim: int = 8):
    return [seed] * dim


def test_add_and_query_returns_nearest_first():
    store = VectorStore()  # ephemeral, in-memory
    chunks = [
        Chunk(text="python expert", source="resume", score=0.0),
        Chunk(text="java developer", source="resume", score=0.0),
    ]
    store.add("store-nearest", chunks, [_emb(0.1), _emb(0.9)])
    results = store.query("store-nearest", _emb(0.1), k=2)  # identical to first chunk
    assert len(results) == 2
    assert results[0].text == "python expert"
    assert results[0].source == "resume"
    assert results[0].score >= results[1].score
    assert results[0].score == 1.0  # exact-match distance 0 -> score 1.0


def test_query_respects_k():
    store = VectorStore()
    chunks = [Chunk(text=f"c{i}", source="resume", score=0.0) for i in range(5)]
    store.add("store-k", chunks, [[float(i)] * 8 for i in range(5)])
    results = store.query("store-k", [0.0] * 8, k=3)
    assert len(results) == 3


def test_namespaces_are_isolated():
    store = VectorStore()
    store.add("store-ns-a", [Chunk(text="alpha", source="resume", score=0.0)], [[0.1] * 8])
    store.add("store-ns-b", [Chunk(text="beta", source="resume", score=0.0)], [[0.1] * 8])
    a = store.query("store-ns-a", [0.1] * 8, k=5)
    assert [c.text for c in a] == ["alpha"]


def test_add_empty_is_noop():
    store = VectorStore()
    store.add("store-empty", [], [])
    assert store.query("store-empty", [0.1] * 8, k=3) == []
