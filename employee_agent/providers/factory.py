from employee_agent.config import Settings, get_settings
from employee_agent.providers.base import Provider
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider
from employee_agent.providers.gemini import GeminiProvider
from employee_agent.providers.ollama_provider import OllamaProvider


def build_provider(settings: Settings | None = None) -> Provider:
    s = settings or get_settings()

    if s.provider == "fake":
        return FakeProvider()

    if s.provider == "ollama":
        return OllamaProvider(model=s.ollama_model, host=s.ollama_host,
                              temperature=s.temperature)

    if s.provider == "gemini":
        if not s.gemini_api_key:
            raise ValueError("gemini_api_key is required when provider='gemini'")
        primary = GeminiProvider(
            api_key=s.gemini_api_key, model=s.gemini_model,
            embedding_model=s.embedding_model, temperature=s.temperature,
        )
        if s.enable_failover:
            fallback = OllamaProvider(model=s.ollama_model, host=s.ollama_host,
                                      temperature=s.temperature)
            return FailoverProvider(primary, fallback)
        return primary

    raise ValueError(f"unknown provider: {s.provider}")
