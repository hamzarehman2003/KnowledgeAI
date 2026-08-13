from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "KnowledgeAI"
    api_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://knowledgeai:change-me@localhost:5432/knowledgeai"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection_name: str = "document_chunks"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_embedding_model: str = "nomic-embed-text"
    chat_temperature: float = 0.1
    embedding_batch_size: int = 64
    max_upload_size_mb: int = 25
    upload_directory: str = "uploads"
    chunk_size_tokens: int = 450
    chunk_overlap_tokens: int = 75


@lru_cache
def get_settings() -> Settings:
    return Settings()
