"""API request and response schemas for retrieval service."""

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Request schema for /retrieve endpoint."""

    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    filters: dict[str, str] | None = Field(
        default=None,
        description="Metadata filters (regulation, domain, risk_category, etc.)",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Minimum similarity score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply reranking",
    )


class ArticleRequest(BaseModel):
    """Request schema for /retrieve-article endpoint."""

    regulation: str = Field(..., description="Regulation name (e.g., 'EU AI Act', 'GDPR')")
    article: str = Field(..., description="Article number (e.g., 'Article 9', 'Article 22')")


class BatchRetrievalRequest(BaseModel):
    """Request schema for /batch-retrieve endpoint."""

    queries: list[str] = Field(..., description="List of search queries", min_length=1, max_length=50)
    top_k: int = Field(default=10, description="Number of results per query", ge=1, le=100)
    filters: dict[str, str] | None = Field(
        default=None,
        description="Metadata filters applied to all queries",
    )


class ChunkMetadata(BaseModel):
    """Metadata for a legal chunk."""

    regulation: str | None
    article: str | None
    paragraph: str | None
    celex: str | None
    domain: str | None
    risk_category: str | None
    effective_date: str | None
    chunk_index: int | None


class Chunk(BaseModel):
    """Legal document chunk."""

    id: str
    score: float | None
    content: str
    metadata: dict
    regulation: str | None = None
    article: str | None = None
    paragraph: str | None = None
    celex: str | None = None
    domain: str | None = None
    risk_category: str | None = None


class RetrievalResponse(BaseModel):
    """Response schema for /retrieve endpoint."""

    success: bool = True
    query: str
    chunks: list[Chunk]
    count: int
    min_score: float
    max_score: float
    avg_score: float
    filters_applied: dict[str, str]


class BatchRetrievalResponse(BaseModel):
    """Response schema for /batch-retrieve endpoint."""

    success: bool = True
    results: list[RetrievalResponse]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    components: dict[str, bool]
    collection_info: dict | None = None


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: str
    error_code: str | None = None
