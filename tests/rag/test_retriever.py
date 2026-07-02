from employee_agent.providers.fake import FakeProvider
from employee_agent.rag.retriever import Retriever
from employee_agent.rag.store import VectorStore
from employee_agent.schemas import Chunk

# Distinct "retr-" namespaces so ephemeral chromadb state never collides with
# test_store.py (see the note in employee_agent/rag/store.py).


def _chunks(texts, source="resume"):
    return [Chunk(text=t, source=source, score=0.0) for t in texts]


async def test_index_then_retrieve_exact_match_ranks_first():
    retr = Retriever(FakeProvider(embed_dim=16), VectorStore())
    await retr.index(
        "retr-exact", _chunks(["python and django", "rust systems", "graphic design"])
    )
    results = await retr.retrieve("retr-exact", "python and django", k=2)
    # identical text -> identical FakeProvider embedding -> distance 0 -> ranks first
    assert results[0].text == "python and django"
    assert results[0].score == 1.0


async def test_retrieve_returns_at_most_k():
    retr = Retriever(FakeProvider(embed_dim=16), VectorStore())
    await retr.index("retr-k", _chunks([f"skill {i}" for i in range(6)]))
    results = await retr.retrieve("retr-k", "skill 0", k=3)
    assert len(results) == 3
    assert all(isinstance(c, Chunk) for c in results)


async def test_index_empty_then_retrieve_is_empty():
    retr = Retriever(FakeProvider(), VectorStore())
    await retr.index("retr-empty", [])
    assert await retr.retrieve("retr-empty", "anything", k=3) == []
