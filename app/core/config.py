from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Storage
    upload_dir: Path = BASE_DIR / "uploads"
    image_extract_dir: Path = BASE_DIR / "extracted_images"
    max_upload_files: int = 20  # cap per POST /documents request (multi-file)

    # Database (SQLAlchemy + Alembic — see alembic/). "development" uses the
    # local SQLite file at db_path; "production" builds a URL from the
    environment: Literal["development", "production"] = "development"
    db_path: Path = BASE_DIR / "rag.db"
    database_url: str | None = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "rag_platform"
    db_user: str = "postgres"
    db_password: str = ""

    # Qdrant (server mode — see docker-compose.yml)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "document_chunks"

    # Embedding models
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_dim: int = 384
    sparse_model: str = "Qdrant/bm25"
    embed_batch_size: int = 32

    # HuggingFace token (for gated models); optional
    hf_token: str | None = None

    # Reranker (sentence-transformers CrossEncoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_enabled: bool = True
    prefetch_limit: int = 20  # candidates fetched from hybrid search before reranking

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # unstructured: "auto" | "hi_res" | "fast" | "ocr_only"
    unstructured_strategy: str = "auto"
    extract_images: bool = True

    # Query
    default_top_k: int = 5
    max_top_k: int = 50

    # Logging
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"
    log_max_bytes: int = 10 * 1024 * 1024  # rotate per file
    log_backup_count: int = 5

    # Rate limiting
    redis_rate_limit: bool = False
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_requests: int = 60  # bucket capacity (max burst)
    rate_limit_window_seconds: int = 60  # capacity refills fully over this window

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.environment == "production":
            return (
                f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        return f"sqlite:///{self.db_path}"
    

    # set true to restore Python 3.13 VERIFY_X509_STRICT (fails behind
    # TLS-inspecting appliances whose CA lacks critical Basic Constraints)
    ssl_strict_verify: bool = False


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for p in (settings.upload_dir, settings.image_extract_dir, settings.log_dir):
        p.mkdir(parents=True, exist_ok=True)
    return settings
