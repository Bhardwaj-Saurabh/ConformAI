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
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API Keys
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Opik Observability (replaces LangSmith)
    opik_api_key: str = Field(default="", alias="OPIK_API_KEY")
    opik_workspace: str = "conformai"
    opik_project: str = "eu-compliance-rag"
    opik_url: str = Field(default="https://www.comet.com/opik/api", alias="OPIK_URL")
    opik_enabled: bool = False

    # Database
    postgres_user: str = "conformai"
    postgres_password: str = "conformai_password"
    postgres_db: str = "conformai"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = Field(
        default="postgresql://conformai:conformai_password@localhost:5432/conformai"
    )

    # Vector Database (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: str = "redis://localhost:6379/0"

    # Embedding Configuration (OpenAI)
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 1024  # Can be 256-3072 for text-embedding-3-large
    embedding_provider: Literal["openai"] = "openai"

    # LLM Configuration
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Retrieval Configuration
    retrieval_top_k: int = 10
    retrieval_fetch_k: int = 50
    retrieval_mmr_lambda: float = 0.7
    retrieval_min_confidence: float = 0.6

    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_article_length: int = 2048

    # Data Sources
    eurlex_api_base_url: str = "https://publications.europa.eu/webapi/rdf/sparql"
    edpb_rss_url: str = "https://edpb.europa.eu/rss.xml"

    # Object Storage (S3/MinIO)
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "conformai-documents"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_workers: int = 4

    # Security
    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = "your_jwt_secret_key_change_me"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # Rate Limiting
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100

    # Monitoring
    prometheus_port: int = 9090
    grafana_port: int = 3000

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
