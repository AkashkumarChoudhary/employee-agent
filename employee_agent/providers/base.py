from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class ProviderError(Exception):
    """Raised when a provider call fails (network, quota, parse)."""


@runtime_checkable
class Provider(Protocol):
    name: str

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
