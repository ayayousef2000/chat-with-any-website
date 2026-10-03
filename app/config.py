"""Application settings loaded from environment variables."""

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embeddings (Cohere)
    cohere_api_key: str
    cohere_embed_model: str = "embed-v5.0-pro"
    cohere_embed_dimension: int = 1024

    # Vector database (Weaviate Cloud)
    weaviate_url: str
    weaviate_api_key: str
    weaviate_collection: str = "WebsiteChunk"

    # LLM (Groq)
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # Reranking (Cohere)
    rerank_enabled: bool = True
    cohere_rerank_model: str = "rerank-v4.0-fast"

    # Fetching
    fetch_timeout_seconds: float = 20.0
    fetch_max_bytes: int = 5_000_000

    # Headless-browser fallback for JavaScript-rendered pages (needs the "browser" extra)
    browser_fallback: bool = False
    min_text_chars: int = Field(default=500, ge=0)

    # Chunking and retrieval
    chunk_size: int = Field(default=2000, gt=0)
    chunk_overlap: int = Field(default=300, ge=0)
    retrieve_k: int = Field(default=25, gt=0)
    top_k: int = Field(default=5, gt=0)
    hybrid_alpha: float = Field(default=0.65, ge=0, le=1)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k > self.retrieve_k:
            raise ValueError("TOP_K must not exceed RETRIEVE_K")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the application settings, loading them once.

    Returns:
        The cached settings instance.

    Raises:
        pydantic.ValidationError: If a required variable is missing or a value is invalid.
    """
    return Settings()  # type: ignore[call-arg]  # required values are read from the environment
