import chromadb
from chromadb.config import Settings

from employee_agent.schemas import Chunk

_SETTINGS = Settings(anonymized_telemetry=False)


class VectorStore:
    """Chroma-backed vector store, one collection per ``namespace``.

    Production uses a persistent client at ``path`` namespaced per unique
    ``job_id``. When ``path`` is ``None`` an in-memory (ephemeral) client is
    used for tests; note chromadb allows only ONE ephemeral instance per
    process, so ephemeral ``VectorStore`` instances share a backing store —
    tests must therefore use distinct namespaces to stay isolated.
    """

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
