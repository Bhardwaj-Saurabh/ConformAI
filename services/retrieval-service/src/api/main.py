"""FastAPI application for retrieval service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.retrieval_service.src.api.schemas import (
    ArticleRequest,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    Chunk,
    ErrorResponse,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from services.retrieval_service.src.retrieval.retriever import get_retrieval_service
from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Retrieval service...")
    logger.info(f"Qdrant URL: {settings.qdrant_url}")
    logger.info(f"Collection: {settings.qdrant_collection_name}")
    logger.info(f"Embedding model: {settings.embedding_model}")

    # Initialize services (singleton pattern ensures single instance)
    retriever = get_retrieval_service()

    yield

    # Shutdown
    logger.info("Shutting down Retrieval service...")


# Create FastAPI app
app = FastAPI(
    title="ConformAI Retrieval Service",
    description="Vector search and retrieval service for EU legal documents",
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
        "service": "ConformAI Retrieval Service",
        "version": "1.0.0",
        "description": "Vector search and retrieval for EU legal documents",
        "endpoints": {
            "retrieve": "/api/v1/retrieve",
            "retrieve_article": "/api/v1/retrieve-article",
            "batch_retrieve": "/api/v1/batch-retrieve",
            "health": "/health",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Checks:
    - Qdrant connection
    - Embedding service
    - Collection availability
    """
    try:
        retriever = get_retrieval_service()

        # Check component health
        health = await retriever.health_check()

        # Get collection info
        collection_info = None
        if health.get("qdrant"):
            try:
                collection_info = await retriever.vector_store.get_collection_info()
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")

        # Determine overall status
        all_healthy = all(health.values())
        status = "healthy" if all_healthy else "degraded"

        return HealthResponse(
            status=status,
            service="retrieval-service",
            version="1.0.0",
            components=health,
            collection_info=collection_info,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="retrieval-service",
            version="1.0.0",
            components={"qdrant": False, "embedder": False},
        )


@app.post("/api/v1/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    Retrieve relevant legal chunks for a query.

    Performs semantic search with optional metadata filtering.

    Args:
        request: Retrieval request with query and parameters

    Returns:
        Retrieved chunks with scores and metadata

    Example:
        POST /api/v1/retrieve
        {
            "query": "What are the obligations for high-risk AI systems?",
            "top_k": 10,
            "filters": {"regulation": "EU AI Act", "risk_category": "high"},
            "score_threshold": 0.6
        }
    """
    logger.info(f"Retrieve request: {request.query[:100]}...")

    try:
        retriever = get_retrieval_service()

        result = await retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            score_threshold=request.score_threshold,
            rerank=request.rerank,
        )

        # Convert to response schema
        return RetrievalResponse(
            query=result["query"],
            chunks=[Chunk(**chunk) for chunk in result["chunks"]],
            count=result["count"],
            min_score=result["min_score"],
            max_score=result["max_score"],
            avg_score=result["avg_score"],
            filters_applied=result["filters_applied"],
        )

    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Retrieval failed: {str(e)}",
                error_code="RETRIEVAL_ERROR",
            ).dict(),
        )


@app.post("/api/v1/retrieve-article")
async def retrieve_article(request: ArticleRequest):
    """
    Retrieve a specific article by regulation name and article number.

    Args:
        request: Article request with regulation and article number

    Returns:
        Article chunk or 404 if not found

    Example:
        POST /api/v1/retrieve-article
        {
            "regulation": "EU AI Act",
            "article": "Article 9"
        }
    """
    logger.info(f"Article request: {request.article} from {request.regulation}")

    try:
        retriever = get_retrieval_service()

        chunk = await retriever.retrieve_by_article(
            regulation=request.regulation,
            article=request.article,
        )

        if chunk:
            return {
                "success": True,
                "chunk": Chunk(**chunk),
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    error=f"Article not found: {request.article} in {request.regulation}",
                    error_code="ARTICLE_NOT_FOUND",
                ).dict(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving article: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Article retrieval failed: {str(e)}",
                error_code="RETRIEVAL_ERROR",
            ).dict(),
        )


@app.post("/api/v1/batch-retrieve", response_model=BatchRetrievalResponse)
async def batch_retrieve(request: BatchRetrievalRequest):
    """
    Retrieve for multiple queries in batch.

    More efficient than making multiple individual requests.

    Args:
        request: Batch retrieval request with list of queries

    Returns:
        List of retrieval results, one per query

    Example:
        POST /api/v1/batch-retrieve
        {
            "queries": [
                "What are high-risk AI obligations?",
                "What are prohibited AI practices?"
            ],
            "top_k": 10,
            "filters": {"regulation": "EU AI Act"}
        }
    """
    logger.info(f"Batch retrieve request: {len(request.queries)} queries")

    try:
        retriever = get_retrieval_service()

        results = await retriever.batch_retrieve(
            queries=request.queries,
            top_k=request.top_k,
            filters=request.filters,
        )

        # Convert to response schema
        response_results = [
            RetrievalResponse(
                query=r["query"],
                chunks=[Chunk(**chunk) for chunk in r["chunks"]],
                count=r["count"],
                min_score=r["min_score"],
                max_score=r["max_score"],
                avg_score=r["avg_score"],
                filters_applied=r["filters_applied"],
            )
            for r in results
        ]

        return BatchRetrievalResponse(results=response_results)

    except Exception as e:
        logger.error(f"Error during batch retrieval: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Batch retrieval failed: {str(e)}",
                error_code="BATCH_RETRIEVAL_ERROR",
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
        port=settings.retrieval_service_port,
        log_level=settings.log_level.lower(),
    )
