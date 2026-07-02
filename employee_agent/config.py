from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-004"
    ollama_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    provider: Literal["gemini", "ollama", "fake"] = "gemini"
    enable_failover: bool = True
    temperature: float = 0.0
    sqlite_path: str = "./data/employee_agent.db"
    chroma_path: str = "./data/chroma"
    api_keys: str = "demo-key"
    rate_limit: str = "100/minute"

    def allowed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
