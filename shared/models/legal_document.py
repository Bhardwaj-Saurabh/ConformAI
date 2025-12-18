"""Legal document data models."""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    """EU AI Act risk categories."""

    PROHIBITED = "prohibited"
    HIGH_RISK = "high"
    LIMITED_RISK = "limited"
    MINIMAL_RISK = "minimal"
    UNCLASSIFIED = "unclassified"


class RegulationType(str, Enum):
    """Types of EU legal documents."""

    REGULATION = "regulation"
    DIRECTIVE = "directive"
    DECISION = "decision"
    GUIDELINE = "guideline"
    OPINION = "opinion"
    RECOMMENDATION = "recommendation"


class AIDomain(str, Enum):
    """AI application domains for classification."""

    BIOMETRICS = "biometrics"
    RECRUITMENT = "recruitment"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    LAW_ENFORCEMENT = "law_enforcement"
    BORDER_CONTROL = "border_control"
    CREDIT_SCORING = "credit_scoring"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    SOCIAL_SCORING = "social_scoring"
    GENERAL = "general"


class Regulation(BaseModel):
    """EU Regulation metadata."""

    celex_id: str = Field(..., description="CELEX number (e.g., 32016R0679)")
    name: str = Field(..., description="Regulation name (e.g., GDPR)")
    full_title: str = Field(..., description="Official full title")
    regulation_type: RegulationType = RegulationType.REGULATION
    adoption_date: date | None = None
    effective_date: date | None = None
    url: str | None = None
    version: str = "consolidated"
    is_active: bool = True


class Chapter(BaseModel):
    """Legal document chapter."""

    number: str = Field(..., description="Chapter number (e.g., 'I', 'II')")
    title: str = Field(..., description="Chapter title")
    articles: list["Article"] = Field(default_factory=list)


class Article(BaseModel):
    """Legal document article."""

    number: str = Field(..., description="Article number")
    title: str | None = None
    content: str = Field(..., description="Full article text")
    paragraphs: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=list, description="References to other articles"
    )


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""

    # Document identifiers
    regulation_name: str = Field(..., description="e.g., 'GDPR', 'AI Act'")
    celex_id: str = Field(..., description="CELEX number")
    regulation_type: RegulationType = RegulationType.REGULATION

    # Structural information
    chapter_number: str | None = None
    chapter_title: str | None = None
    article_number: str = Field(..., description="Article number")
    article_title: str | None = None
    paragraph_index: int | None = None

    # Chunk information
    chunk_index: int = Field(default=0, description="Index of this chunk within article")
    total_chunks: int = Field(default=1, description="Total chunks for this article")

    # Temporal information
    effective_date: date | None = None
    version: str = "consolidated"

    # Classification
    domains: list[AIDomain] = Field(
        default_factory=list, description="Relevant AI domains"
    )
    risk_category: RiskCategory | None = None

    # Source
    source_url: str | None = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """Document chunk for embedding and retrieval."""

    text: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata
    embedding: list[float] | None = Field(
        default=None, description="Vector embedding"
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }


class LegalDocument(BaseModel):
    """Complete legal document structure."""

    regulation: Regulation
    chapters: list[Chapter] = Field(default_factory=list)
    annexes: dict[str, str] = Field(default_factory=dict)
    raw_text: str | None = None
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class QueryClassification(BaseModel):
    """Classification of user query for optimized retrieval."""

    # Query understanding
    original_query: str
    intent: str = Field(..., description="Detected intent (e.g., 'compliance_check')")

    # Legal scope
    regulations: list[str] = Field(
        default_factory=list, description="Relevant regulations (e.g., ['GDPR', 'AI Act'])"
    )
    explicit_articles: list[str] = Field(
        default_factory=list, description="Explicitly mentioned articles"
    )

    # Domain classification
    domain: AIDomain = AIDomain.GENERAL
    subdomains: list[str] = Field(default_factory=list)

    # Risk assessment
    risk_category: RiskCategory | None = None

    # Question type
    question_type: str = Field(
        default="general",
        description="Type: 'compliance', 'obligation', 'prohibition', 'definition'",
    )

    # Metadata
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    complexity_score: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Query complexity"
    )
    requires_multi_regulation: bool = False

    # Suggested filters for retrieval
    suggested_filters: dict[str, Any] = Field(default_factory=dict)


# Update forward references
Chapter.model_rebuild()
Article.model_rebuild()
