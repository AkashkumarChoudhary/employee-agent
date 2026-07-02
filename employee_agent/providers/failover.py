from employee_agent.providers.base import BaseModelT, Provider, ProviderError


class FailoverProvider:
    def __init__(self, primary: Provider, fallback: Provider):
        self._primary = primary
        self._fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            return await self._primary.generate_structured(
                system=system, prompt=prompt, schema=schema
            )
        except ProviderError:
            return await self._fallback.generate_structured(
                system=system, prompt=prompt, schema=schema
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._primary.embed(texts)
        except ProviderError:
            return await self._fallback.embed(texts)
