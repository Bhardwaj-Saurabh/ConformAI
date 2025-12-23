"""API request and response schemas."""

from graph.state import Citation
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for /query endpoint."""

    query: str = Field(..., description="User's compliance question", min_length=10)
    conversation_id: str | None = Field(
        default=None, description="Optional conversation tracking ID"
    )
    user_id: str | None = Field(
        default=None, description="Optional user identifier for memory persistence"
    )
    filters: dict[str, str] | None = Field(
        default=None,
        description="Optional metadata filters (regulation, domain, risk_category)",
    )
    max_iterations: int = Field(
        default=5, description="Maximum ReAct agent iterations", ge=1, le=10
    )


class QueryMetadata(BaseModel):
    """Metadata about query processing."""

    intent: str
    ai_domain: str | None
    risk_category: str | None
    query_complexity: str
    processing_time_ms: float
    total_llm_calls: int
    total_tokens_used: int
    confidence_score: float
    agent_iterations: int
    retrieval_count: int


class ReasoningStep(BaseModel):
    """Agent reasoning step."""

    step: int
    thought: str
    action: str
    observation: str | None


class QueryResponse(BaseModel):
    """Response schema for /query endpoint."""

    success: bool
    query: str
    answer: str
    citations: list[Citation]
    metadata: QueryMetadata
    reasoning_trace: list[str] | None = Field(
        default=None, description="Agent's reasoning steps"
    )
    agent_actions: list[ReasoningStep] | None = Field(
        default=None, description="Detailed agent actions"
    )
    refusal_reason: str | None = Field(
        default=None, description="Reason for refusal if query was rejected"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = False
    error: str
    error_code: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    llm_provider: str
    llm_model: str
