"""
ConformAI API Gateway

Main FastAPI application for EU AI compliance intelligence.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared.config import get_settings
from shared.utils import get_logger

# Initialize settings and logger
settings = get_settings()
logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting ConformAI API Gateway...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Vector DB: {settings.qdrant_url}")

    yield

    logger.info("Shutting down ConformAI API Gateway...")


# Create FastAPI app
app = FastAPI(
    title="ConformAI API",
    description="RAG-based EU AI Compliance Intelligence System",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "ConformAI API Gateway",
        "version": "0.1.0",
        "status": "operational",
        "environment": settings.environment,
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "api_gateway": "operational",
            # TODO: Add health checks for other services
            "rag_service": "unknown",
            "retrieval_service": "unknown",
            "vector_db": "unknown",
            "postgres": "unknown",
        },
    }


# API v1 router (to be implemented)
@app.get("/api/v1/status")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def api_status(request: Request):
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "operational",
        "rate_limits": {
            "per_minute": settings.rate_limit_per_minute,
            "per_hour": settings.rate_limit_per_hour,
        },
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "detail": str(exc) if settings.is_development else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and settings.is_development,
        log_level=settings.log_level.lower(),
    )
