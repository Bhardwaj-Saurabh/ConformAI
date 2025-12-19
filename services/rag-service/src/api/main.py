"""FastAPI application for RAG service."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.rag_service.src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
)
from services.rag_service.src.graph.graph import run_rag_pipeline
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting RAG service...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")

    yield

    # Shutdown
    logger.info("Shutting down RAG service...")


# Create FastAPI app
app = FastAPI(
    title="ConformAI RAG Service",
    description="Agentic RAG service for EU AI compliance queries with ReAct agent and query decomposition",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Routes =====


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ConformAI RAG Service",
        "version": "1.0.0",
        "description": "Agentic RAG for EU AI compliance queries",
        "endpoints": {
            "query": "/api/v1/query",
            "health": "/health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="rag-service",
        version="1.0.0",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_compliance(request: QueryRequest):
    """
    Process EU AI compliance query using agentic RAG pipeline.

    This endpoint:
    1. Analyzes and decomposes complex queries
    2. Uses ReAct agent to iteratively retrieve and answer sub-questions
    3. Synthesizes final answer with citations
    4. Validates grounding to prevent hallucinations

    Args:
        request: Query request with question and optional parameters

    Returns:
        Query response with answer, citations, and metadata

    Example:
        POST /api/v1/query
        {
            "query": "What are the documentation requirements for high-risk AI systems in recruitment?",
            "max_iterations": 5
        }
    """
    start_time = time.time()

    logger.info(f"Received query: {request.query[:100]}...")

    try:
        # Update max iterations if provided
        from services.rag_service.src.graph.state import create_initial_state

        initial_state = create_initial_state(request.query, request.conversation_id)
        initial_state["max_iterations"] = request.max_iterations

        # Run RAG pipeline
        result = await run_rag_pipeline(request.query, request.conversation_id)

        # Check if refused
        if result.get("refusal_reason"):
            return QueryResponse(
                success=False,
                query=request.query,
                answer="",
                citations=[],
                metadata=QueryMetadata(
                    intent=result.get("intent", "unknown"),
                    ai_domain=str(result.get("ai_domain")) if result.get("ai_domain") else None,
                    risk_category=str(result.get("risk_category")) if result.get("risk_category") else None,
                    query_complexity=result.get("query_complexity", "unknown"),
                    processing_time_ms=result.get("processing_time_ms", 0),
                    total_llm_calls=result.get("total_llm_calls", 0),
                    total_tokens_used=result.get("total_tokens_used", 0),
                    confidence_score=0.0,
                    agent_iterations=result.get("iteration_count", 0),
                    retrieval_count=len(result.get("all_retrieved_chunks", [])),
                ),
                refusal_reason=result["refusal_reason"],
            )

        # Build response
        response = QueryResponse(
            success=True,
            query=request.query,
            answer=result.get("final_answer", ""),
            citations=result.get("citations", []),
            metadata=QueryMetadata(
                intent=result.get("intent", "unknown"),
                ai_domain=str(result.get("ai_domain")) if result.get("ai_domain") else None,
                risk_category=str(result.get("risk_category")) if result.get("risk_category") else None,
                query_complexity=result.get("query_complexity", "unknown"),
                processing_time_ms=result.get("processing_time_ms", 0),
                total_llm_calls=result.get("total_llm_calls", 0),
                total_tokens_used=result.get("total_tokens_used", 0),
                confidence_score=result.get("confidence_score", 0.0),
                agent_iterations=result.get("iteration_count", 0),
                retrieval_count=len(result.get("all_retrieved_chunks", [])),
            ),
            reasoning_trace=result.get("reasoning_trace", []),
            agent_actions=[
                ReasoningStep(
                    step=action.step,
                    thought=action.thought,
                    action=action.action,
                    observation=action.observation,
                )
                for action in result.get("agent_actions", [])
            ],
        )

        logger.info(
            f"Query processed successfully in {result.get('processing_time_ms', 0):.0f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Internal server error: {str(e)}",
                error_code="INTERNAL_ERROR",
            ).dict(),
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.rag_service_port,
        log_level=settings.log_level.lower(),
    )
