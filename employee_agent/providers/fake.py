import hashlib

from pydantic import BaseModel

from employee_agent.providers.base import BaseModelT, ProviderError


def _default_instance(schema: type[BaseModel]) -> BaseModel:
    """Construct a schema instance from each field's type default."""
    values = {}
    for fname, field in schema.model_fields.items():
        ann = field.annotation
        if ann in (int, float):
            values[fname] = 0
        elif ann is bool:
            values[fname] = False
        elif ann is str:
            values[fname] = ""
        elif ann is list or getattr(ann, "__origin__", None) is list:
            values[fname] = []
        else:
            values[fname] = None
    return schema.model_construct(**values)


class FakeProvider:
    name = "fake"

    def __init__(self, responses=None, embed_dim: int = 8, fail: bool = False):
        self._responses = responses or {}
        self._embed_dim = embed_dim
        self._fail = fail

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        if self._fail:
            raise ProviderError("fake failure")
        if schema in self._responses:
            return self._responses[schema]
        return _default_instance(schema)  # type: ignore[return-value]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise ProviderError("fake failure")
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self._embed_dim)])
        return out
