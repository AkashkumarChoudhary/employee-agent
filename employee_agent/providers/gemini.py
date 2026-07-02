from google import genai
from google.genai import types

from employee_agent.providers.base import BaseModelT, ProviderError


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, embedding_model: str,
                 temperature: float = 0.0):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model
        self._temperature = temperature

    async def generate_structured(
        self, *, system: str, prompt: str, schema: type[BaseModelT]
    ) -> BaseModelT:
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self._temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return schema.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001 - normalize all SDK/parse errors
            raise ProviderError(f"gemini generate failed: {e}") from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.aio.models.embed_content(
                model=self._embedding_model, contents=texts
            )
            return [list(e.values) for e in resp.embeddings]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"gemini embed failed: {e}") from e
