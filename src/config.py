"""Central configuration loaded from .env via pydantic-settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    voyage_api_key: str
    anthropic_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # Embeddings
    voyage_model: str = "voyage-3"
    embedding_dim: int = 1024

    # Generation
    anthropic_model: str = "claude-sonnet-4-6"

    # Chunking & retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
