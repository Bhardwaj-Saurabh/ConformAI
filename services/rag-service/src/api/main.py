"""FastAPI application for RAG service."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
)
from graph.graph import run_rag_pipeline
from shared.config.settings import get_settings
from shared.utils.logger import get_logger, set_request_context, clear_request_context
from shared.utils.opik_tracer import get_opik_client, log_metric, log_event

# Import health check router
try:
    from api.health import router as health_router
    HEALTH_ROUTER_AVAILABLE = True
except ImportError:
    HEALTH_ROUTER_AVAILABLE = False
    logger.warning("Health router not available, using basic health check")

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting RAG service...")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"LLM Model: {settings.llm_model}")

    # Initialize Opik client
    if settings.opik_enabled:
        opik_client = get_opik_client()
        if opik_client:
            logger.info("✓ Opik observability enabled")
            log_event("rag_service_started", {
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "environment": settings.environment,
            })
        else:
            logger.warning("⚠ Opik observability disabled (initialization failed)")
    else:
        logger.info("Opik observability disabled")

    yield

    # Shutdown
    logger.info("Shutting down RAG service...")
    if settings.opik_enabled:
        log_event("rag_service_shutdown", {"environment": settings.environment})


# Create FastAPI app
app = FastAPI(
    title="ConformAI RAG Service",
    description="Agentic RAG service for EU AI compliance queries with ReAct agent and query decomposition",
    version="1.0.0",
    lifespan=lifespan,
)

# Add request ID middleware for tracking and logging
@app.middleware("http")
async def add_request_id_and_logging(request, call_next):
    """Add request ID to all requests for tracking and set logging context."""
    import uuid

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    # Set request context for logging
    set_request_context(request_id=request_id)

    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "http_method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None,
        }
    )

    start_time = time.time()

    try:
        response = await call_next(request)

        # Calculate request duration
        duration_ms = (time.time() - start_time) * 1000

        # Log request completion
        logger.log_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id

        return response

    finally:
        # Clear request context
        clear_request_context()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include health check router if available
if HEALTH_ROUTER_AVAILABLE:
    app.include_router(health_router)
    logger.info("Comprehensive health check endpoints enabled")


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

    # Set conversation context if provided
    if request.conversation_id:
        set_request_context(conversation_id=request.conversation_id)

    logger.info(
        f"Processing compliance query",
        extra={
            "query_length": len(request.query),
            "query_preview": request.query[:100],
            "max_iterations": request.max_iterations,
            "has_conversation_id": bool(request.conversation_id),
        }
    )

    # Log audit event
    logger.log_audit(
        action="query_received",
        resource="rag_pipeline",
        result="processing",
        query_length=len(request.query),
        max_iterations=request.max_iterations,
    )

    # Log request event to Opik
    log_event("query_received", {
        "query_length": len(request.query),
        "max_iterations": request.max_iterations,
        "has_conversation_id": bool(request.conversation_id),
    })

    try:
        # Update max iterations if provided
        from graph.state import create_initial_state

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

        api_duration_ms = (time.time() - start_time) * 1000

        # Log successful completion with performance metrics
        logger.log_performance(
            operation="query_compliance",
            duration_ms=api_duration_ms,
            pipeline_duration_ms=result.get('processing_time_ms', 0),
            confidence_score=result.get("confidence_score", 0.0),
            iterations=result.get("iteration_count", 0),
            llm_calls=result.get("total_llm_calls", 0),
            tokens_used=result.get("total_tokens_used", 0),
            citations_count=len(result.get("citations", [])),
        )

        # Log audit event
        logger.log_audit(
            action="query_completed",
            resource="rag_pipeline",
            result="success",
            confidence_score=result.get("confidence_score", 0.0),
            iterations=result.get("iteration_count", 0),
            answer_length=len(result.get("final_answer", "")),
        )

        # Log metrics to Opik
        log_metric("api_request_duration_ms", api_duration_ms, {
            "endpoint": "query",
            "success": "true",
        })
        log_event("query_completed", {
            "success": True,
            "duration_ms": api_duration_ms,
            "confidence_score": result.get("confidence_score", 0.0),
            "iterations": result.get("iteration_count", 0),
        })

        logger.info(
            f"✓ Query processed successfully",
            extra={
                "total_duration_ms": api_duration_ms,
                "confidence_score": result.get("confidence_score", 0.0),
            }
        )

        return response

    except Exception as e:
        api_duration_ms = (time.time() - start_time) * 1000

        # Log error with full context
        logger.log_error_with_context(
            message="Failed to process compliance query",
            error=e,
            query_length=len(request.query),
            max_iterations=request.max_iterations,
            duration_ms=api_duration_ms,
        )

        # Log audit event for failure
        logger.log_audit(
            action="query_failed",
            resource="rag_pipeline",
            result="error",
            error_type=type(e).__name__,
            error_message=str(e),
        )

        # Log error metrics to Opik
        log_metric("api_request_duration_ms", api_duration_ms, {
            "endpoint": "query",
            "success": "false",
            "error": "true",
        })
        log_event("query_failed", {
            "error": str(e),
            "duration_ms": api_duration_ms,
        })

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
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
            "timestamp": time.time(),
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
