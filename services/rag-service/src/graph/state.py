"""LangGraph state schema for agentic RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from shared.models.legal_document import AIDomain, Chunk, RiskCategory


@dataclass
class SubQuery:
    """Decomposed sub-question for complex queries."""

    question: str
    aspect: str  # "obligations", "prohibitions", "risk_classification", etc.
    priority: int  # 1=critical, 2=important, 3=supplementary
    answer: str | None = None
    sources: list[Chunk] = field(default_factory=list)
    status: Literal["pending", "in_progress", "completed"] = "pending"


@dataclass
class Citation:
    """Citation reference to legal source."""

    source_id: int
    regulation: str
    article: str | None
    paragraph: str | None
    celex: str | None
    excerpt: str
    chunk_id: str


@dataclass
class AgentAction:
    """ReAct agent action with thought, action, and observation."""

    step: int
    thought: str  # Agent's reasoning process
    action: str  # Tool name to execute
    action_input: dict[str, Any]  # Tool parameters
    observation: str | None = None  # Tool execution result
    timestamp: float = 0.0
    error: str | None = None  # Error if action failed


class RAGState(TypedDict, total=False):
    """
    LangGraph state for agentic RAG pipeline.

    This state flows through all nodes in the graph and accumulates
    information at each step of query processing.
    """

    # ===== Input =====
    query: str  # User's original question
    conversation_id: str | None  # Optional conversation tracking
    user_id: str | None  # User identifier for memory
    user_context: dict[str, Any] | None  # Additional user-provided context

    # ===== Memory & Context =====
    conversation_history: list[dict[str, Any]]  # Previous messages in conversation
    conversation_context_summary: str  # Summary of recent conversation
    user_memories: list[dict[str, Any]]  # Long-term user memories
    user_profile: dict[str, Any]  # Structured user profile from memories
    user_context_summary: str  # Summary of user facts/preferences

    # ===== Query Analysis =====
    intent: str  # compliance_question, risk_assessment, obligation_lookup, etc.
    ai_domain: AIDomain | None  # biometrics, recruitment, healthcare, etc.
    risk_category: RiskCategory | None  # prohibited, high, limited, minimal
    entities: list[str]  # Extracted entities (regulations, articles, systems)
    query_complexity: Literal["simple", "medium", "complex"]

    # ===== Query Decomposition =====
    sub_queries: list[SubQuery]  # Decomposed sub-questions
    decomposition_needed: bool  # Whether decomposition was required

    # ===== ReAct Agent Loop =====
    agent_actions: list[AgentAction]  # History of all agent steps
    agent_state: Literal["planning", "acting", "observing", "done"]
    iteration_count: int  # Current iteration in loop
    max_iterations: int  # Maximum allowed iterations (default: 5)
    working_memory: dict[str, Any]  # Agent's scratchpad for intermediate state

    # ===== Retrieval =====
    all_retrieved_chunks: list[Chunk]  # All chunks across all retrievals
    retrieval_history: list[dict[str, Any]]  # Track all retrieval calls
    retrieval_scores: list[float]  # Relevance scores
    min_score: float  # Minimum retrieval score
    max_score: float  # Maximum retrieval score

    # ===== Generation =====
    intermediate_answers: dict[str, str]  # Sub-query -> answer mapping
    final_answer: str  # Synthesized final answer
    reasoning_trace: list[str]  # Explanation of reasoning steps

    # ===== Citations & Grounding =====
    citations: list[Citation]  # All citations in final answer
    grounding_validated: bool  # Whether answer passed grounding check
    hallucination_detected: bool  # Whether hallucinations were found

    # ===== Safety & Validation =====
    is_safe: bool  # Whether query passed safety checks
    refusal_reason: str | None  # Reason for refusal if rejected
    confidence_score: float  # Overall confidence in answer

    # ===== Metadata =====
    processing_time_ms: float  # Total processing time
    model_used: str  # Primary LLM model used
    total_llm_calls: int  # Number of LLM calls made
    total_tokens_used: int  # Total tokens consumed

    # ===== Error Handling =====
    error: str | None  # Error message if something failed
    retry_count: int  # Number of retries attempted


def create_initial_state(
    query: str, conversation_id: str | None = None, user_id: str | None = None
) -> RAGState:
    """
    Create initial RAG state from user query.

    Args:
        query: User's question
        conversation_id: Optional conversation tracking ID
        user_id: Optional user identifier for memory

    Returns:
        Initialized RAGState with defaults
    """
    return RAGState(
        # Input
        query=query,
        conversation_id=conversation_id,
        user_id=user_id,
        user_context=None,
        # Memory & Context
        conversation_history=[],
        conversation_context_summary="",
        user_memories=[],
        user_profile={},
        user_context_summary="",
        # Query Analysis
        intent="",
        ai_domain=None,
        risk_category=None,
        entities=[],
        query_complexity="simple",
        # Decomposition
        sub_queries=[],
        decomposition_needed=False,
        # Agent
        agent_actions=[],
        agent_state="planning",
        iteration_count=0,
        max_iterations=5,  # Default max iterations
        working_memory={},
        # Retrieval
        all_retrieved_chunks=[],
        retrieval_history=[],
        retrieval_scores=[],
        min_score=0.0,
        max_score=0.0,
        # Generation
        intermediate_answers={},
        final_answer="",
        reasoning_trace=[],
        # Citations
        citations=[],
        grounding_validated=False,
        hallucination_detected=False,
        # Safety
        is_safe=True,
        refusal_reason=None,
        confidence_score=0.0,
        # Metadata
        processing_time_ms=0.0,
        model_used="",
        total_llm_calls=0,
        total_tokens_used=0,
        # Error handling
        error=None,
        retry_count=0,
    )
