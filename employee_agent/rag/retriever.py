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
