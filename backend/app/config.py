"""Centralized settings loaded from environment variables (.env)."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/semantic_search"
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_api_key: str | None = None
    use_openai_embeddings: bool = False
    cors_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
