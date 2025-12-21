"""FastAPI application for RAG service."""

import time
from contextlib import asynccontextmanager

from api.schemas import (
    ErrorResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    ReasoningStep,
)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from graph.graph import run_rag_pipeline

from shared.config.settings import get_settings
from shared.utils.logger import clear_request_context, get_logger, set_request_context
from shared.utils.opik_tracer import get_opik_client, log_event, log_metric

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


# ===== Global Error Handlers =====


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with structured error response."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle ValueError exceptions (invalid input)."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.log_error_with_context(
        message="Invalid input value",
        error=exc,
        path=request.url.path,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Global exception handler for uncaught exceptions.

    Logs error with full context and returns structured error response.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.log_error_with_context(
        message="Unhandled exception in request processing",
        error=exc,
        path=request.url.path,
        method=request.method,
        request_id=request_id,
    )

    # Log to Opik if enabled
    if settings.opik_enabled:
        try:
            log_event("rag_service_error", {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_id": request_id,
                "path": request.url.path,
            })
        except Exception as opik_error:
            logger.warning(f"Failed to log error to Opik: {opik_error}")

    # Return generic error to client (don't expose internal details)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred while processing your request",
            "request_id": request_id,
            "timestamp": time.time(),
        },
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


# Note: Comprehensive health check endpoints are provided by health_router
# Includes: /health, /health/ready, /health/live, /health/startup, /metrics


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
        "Processing compliance query",
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
            "✓ Query processed successfully",
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


@app.post("/api/v1/query/stream")
async def query_compliance_stream(request: QueryRequest):
    """
    Stream EU AI compliance query responses using Server-Sent Events (SSE).

    This endpoint provides real-time streaming of the RAG pipeline execution:
    1. Streams analysis progress
    2. Streams retrieval updates
    3. Streams answer chunks as they're generated
    4. Streams final citations and metadata

    Args:
        request: Query request with question and optional parameters

    Returns:
        Server-Sent Events stream with JSON payloads

    Example:
        POST /api/v1/query/stream
        {
            "query": "What are the requirements for high-risk AI systems?",
            "max_iterations": 5
        }

        Response (SSE format):
        data: {"type": "status", "message": "Analyzing query..."}

        data: {"type": "status", "message": "Retrieving documents..."}

        data: {"type": "chunk", "content": "High-risk AI systems..."}

        data: {"type": "done", "metadata": {...}}
    """
    import json
    import asyncio

    async def event_generator():
        """Generate SSE events for the query processing."""
        try:
            # Set conversation context
            if request.conversation_id:
                set_request_context(conversation_id=request.conversation_id)

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting RAG pipeline...'})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for client connection

            # Send analysis status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing query intent and complexity...'})}\n\n"
            await asyncio.sleep(0.2)

            # Run RAG pipeline (non-streaming for now, will enhance later)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Running ReAct agent to retrieve relevant legal sources...'})}\n\n"

            # Execute pipeline
            result = await run_rag_pipeline(request.query, request.conversation_id)

            # Stream the answer in chunks
            if result.get("final_answer"):
                answer = result["final_answer"]

                # Send answer in chunks (simulate streaming)
                chunk_size = 50  # Characters per chunk
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.05)  # Small delay between chunks for effect

            # Send citations
            if result.get("citations"):
                yield f"data: {json.dumps({'type': 'citations', 'citations': [c.dict() if hasattr(c, 'dict') else c for c in result['citations']]})}\n\n"

            # Send final metadata
            metadata = {
                "confidence_score": result.get("confidence_score", 0.0),
                "processing_time_ms": result.get("processing_time_ms", 0),
                "iterations": result.get("iteration_count", 0),
                "chunks_retrieved": len(result.get("all_retrieved_chunks", [])),
                "success": True,
                "refusal_reason": result.get("refusal_reason"),
            }
            yield f"data: {json.dumps({'type': 'done', 'metadata': metadata})}\n\n"

            logger.info("✓ Streaming query completed successfully")

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "error": str(e),
                "error_code": "INTERNAL_ERROR"
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
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
