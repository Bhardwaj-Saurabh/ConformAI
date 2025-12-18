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

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Opik Observability (replaces LangSmith)
    opik_api_key: str = Field(default="", alias="OPIK_API_KEY")
    opik_workspace: str = Field(default="conformai", alias="COMET_WORKSPACE")
    opik_project: str = Field(default="eu-compliance-rag", alias="OPIK_PROJECT_NAME")
    opik_url: str = Field(default="https://www.comet.com/opik/api", alias="OPIK_URL")
    opik_enabled: bool = Field(default=False, alias="OPIK_ENABLED")

    # Database
    postgres_user: str = Field(default="conformai", alias="POSTGRES_USER")
    postgres_password: str = Field(
        default="conformai_password", alias="POSTGRES_PASSWORD"
    )
    postgres_db: str = Field(default="conformai", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    database_url: str = Field(
        default="postgresql://conformai:conformai_password@localhost:5432/conformai",
        alias="DATABASE_URL",
    )

    # Vector Database (Qdrant)
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # Embedding Configuration (OpenAI)
    embedding_model: str = Field(default="text-embedding-3-large", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(
        default=1024, alias="EMBEDDING_DIMENSION"
    )  # Can be 256-3072 for text-embedding-3-large
    embedding_provider: Literal["openai"] = Field(default="openai", alias="EMBEDDING_PROVIDER")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # LLM Configuration
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    retrieval_fetch_k: int = Field(default=50, alias="RETRIEVAL_FETCH_K")
    retrieval_mmr_lambda: float = Field(default=0.7, alias="RETRIEVAL_MMR_LAMBDA")
    retrieval_min_confidence: float = Field(
        default=0.6, alias="RETRIEVAL_MIN_CONFIDENCE"
    )

    # Document Processing
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    max_article_length: int = Field(default=2048, alias="MAX_ARTICLE_LENGTH")

    # Data Sources
    eurlex_api_base_url: str = Field(
        default="https://publications.europa.eu/webapi/rdf/sparql",
        alias="EURLEX_API_BASE_URL",
    )
    edpb_rss_url: str = Field(default="https://edpb.europa.eu/rss.xml", alias="EDPB_RSS_URL")

    # Object Storage (S3/MinIO)
    s3_endpoint_url: str = Field(default="http://localhost:9000", alias="S3_ENDPOINT_URL")
    s3_access_key: str = Field(default="minioadmin", alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin", alias="S3_SECRET_KEY")
    s3_bucket_name: str = Field(default="conformai-documents", alias="S3_BUCKET_NAME")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")
    api_workers: int = Field(default=4, alias="API_WORKERS")
    rag_service_host: str = Field(default="localhost", alias="RAG_SERVICE_HOST")
    rag_service_port: int = Field(default=8001, alias="RAG_SERVICE_PORT")
    retrieval_service_host: str = Field(default="localhost", alias="RETRIEVAL_SERVICE_HOST")
    retrieval_service_port: int = Field(default=8002, alias="RETRIEVAL_SERVICE_PORT")

    # Security
    api_key_header: str = Field(default="X-API-Key", alias="API_KEY_HEADER")
    jwt_secret_key: str = Field(
        default="your_jwt_secret_key_change_me", alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(
        default=60, alias="JWT_EXPIRATION_MINUTES"
    )

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=10, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(default=100, alias="RATE_LIMIT_PER_HOUR")

    # Monitoring
    prometheus_port: int = Field(default=9090, alias="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, alias="GRAFANA_PORT")

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
