import ollama

from employee_agent.providers.base import BaseModelT, ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str, host: str, embedding_model: str | None = None,
                 temperature: float = 0.0):
        self._client = ollama.AsyncClient(host=host)
        self._model = model
        self._embedding_model = embedding_model or model
        self._temperature = temperature

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            resp = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                format=schema.model_json_schema(),
                options={"temperature": self._temperature},
            )
            return schema.model_validate_json(resp["message"]["content"])
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"ollama generate failed: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.embed(model=self._embedding_model, input=texts)
            return [list(v) for v in resp["embeddings"]]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"ollama embed failed: {e}") from e
