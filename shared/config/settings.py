"""Application settings and configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # API Keys (Required - must be set in .env)
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # Opik Observability (Optional - loaded from .env)
    opik_api_key: str | None = Field(default=None, alias="OPIK_API_KEY")
    opik_workspace: str = Field(alias="COMET_WORKSPACE")
    opik_project: str = Field(alias="OPIK_PROJECT_NAME")
    opik_url: str = Field(alias="OPIK_URL")
    opik_enabled: bool = Field(alias="OPIK_ENABLED")

    # Database (Loaded from .env)
    postgres_user: str = Field(alias="POSTGRES_USER")
    postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_host: str = Field(alias="POSTGRES_HOST")
    postgres_port: int = Field(alias="POSTGRES_PORT")
    database_url: str = Field(alias="DATABASE_URL")

    # Vector Database (Qdrant - Loaded from .env)
    qdrant_host: str = Field(alias="QDRANT_HOST")
    qdrant_port: int = Field(alias="QDRANT_PORT")
    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Redis (Loaded from .env)
    redis_host: str = Field(alias="REDIS_HOST")
    redis_port: int = Field(alias="REDIS_PORT")
    redis_db: int = Field(alias="REDIS_DB")
    redis_url: str = Field(alias="REDIS_URL")

    # Embedding Configuration (Loaded from .env)
    embedding_model: str = Field(alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(alias="EMBEDDING_DIMENSION")  # 256-3072 for text-embedding-3-large
    embedding_provider: Literal["openai"] = Field(alias="EMBEDDING_PROVIDER")
    embedding_device: str = Field(alias="EMBEDDING_DEVICE")

    # LLM Configuration (Loaded from .env)
    llm_provider: Literal["anthropic", "openai"] = Field(alias="LLM_PROVIDER")
    llm_model: str = Field(alias="LLM_MODEL")
    llm_temperature: float = Field(alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(alias="LLM_MAX_TOKENS")

    # Retrieval Configuration (Loaded from .env)
    retrieval_top_k: int = Field(alias="RETRIEVAL_TOP_K")
    retrieval_fetch_k: int = Field(alias="RETRIEVAL_FETCH_K")
    retrieval_mmr_lambda: float = Field(alias="RETRIEVAL_MMR_LAMBDA")
    retrieval_min_confidence: float = Field(alias="RETRIEVAL_MIN_CONFIDENCE")

    # Document Processing (Loaded from .env)
    chunk_size: int = Field(alias="CHUNK_SIZE")
    chunk_overlap: int = Field(alias="CHUNK_OVERLAP")
    max_article_length: int = Field(alias="MAX_ARTICLE_LENGTH")

    # Data Sources (Loaded from .env)
    eurlex_api_base_url: str = Field(alias="EURLEX_API_BASE_URL")
    edpb_rss_url: str = Field(alias="EDPB_RSS_URL")

    # Object Storage (S3/MinIO - Loaded from .env)
    s3_endpoint_url: str = Field(alias="S3_ENDPOINT_URL")
    s3_access_key: str = Field(alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(alias="S3_SECRET_KEY")
    s3_bucket_name: str = Field(alias="S3_BUCKET_NAME")

    # API Configuration (Loaded from .env)
    api_host: str = Field(alias="API_HOST")
    api_port: int = Field(alias="API_PORT")
    api_reload: bool = Field(alias="API_RELOAD")
    api_workers: int = Field(alias="API_WORKERS")
    rag_service_host: str = Field(alias="RAG_SERVICE_HOST")
    rag_service_port: int = Field(alias="RAG_SERVICE_PORT")
    retrieval_service_host: str = Field(alias="RETRIEVAL_SERVICE_HOST")
    retrieval_service_port: int = Field(alias="RETRIEVAL_SERVICE_PORT")

    # Security (Loaded from .env)
    api_key_header: str = Field(alias="API_KEY_HEADER")
    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(alias="JWT_EXPIRATION_MINUTES")

    # Rate Limiting (Loaded from .env)
    rate_limit_per_minute: int = Field(alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(alias="RATE_LIMIT_PER_HOUR")

    # Monitoring (Loaded from .env)
    prometheus_port: int = Field(alias="PROMETHEUS_PORT")
    grafana_port: int = Field(alias="GRAFANA_PORT")

    @property
    def qdrant_collection_name(self) -> str:
        """Get Qdrant collection name based on environment."""
        return f"eu_legal_documents_{self.environment}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
