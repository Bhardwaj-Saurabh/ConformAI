"""Health check endpoints for RAG service."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

router = APIRouter(tags=["Health"])
settings = get_settings()
logger = get_logger(__name__)


@router.get("/health", summary="Comprehensive health check")
async def health_check() -> dict[str, Any]:
    """
    Comprehensive health check for RAG service.

    Checks:
    - Qdrant connectivity and document count
    - OpenAI API availability
    - System resources

    Returns:
        Health status with component details
    """
    components = {}
    overall_healthy = True

    # Check Qdrant
    try:
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        retrieval_path = Path(__file__).parent.parent.parent.parent / "retrieval-service" / "src"
        sys.path.insert(0, str(retrieval_path))

        from retrieval.qdrant_client import get_qdrant_store

        store = get_qdrant_store()
        qdrant_healthy = await store.health_check()

        if qdrant_healthy:
            info = await store.get_collection_info()
            components["qdrant"] = {
                "status": "healthy",
                "collection": info.get("name"),
                "documents_indexed": info.get("points_count", 0),
            }
        else:
            overall_healthy = False
            components["qdrant"] = {"status": "unhealthy", "error": "Connection failed"}

    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        overall_healthy = False
        components["qdrant"] = {"status": "unhealthy", "error": str(e)}

    # Check OpenAI API
    try:
        from llm.client import get_llm_client

        llm = get_llm_client()
        # Just check if client is initialized
        components["llm"] = {
            "status": "healthy",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        overall_healthy = False
        components["llm"] = {"status": "unhealthy", "error": str(e)}

    # System info
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        components["system"] = {
            "status": "healthy" if cpu < 90 and memory.percent < 90 else "degraded",
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(memory.percent, 1),
            "memory_available_gb": round(memory.available / (1024**3), 2),
        }

        if cpu >= 90 or memory.percent >= 90:
            overall_healthy = False

    except Exception as e:
        logger.error(f"System health check failed: {e}")
        components["system"] = {"status": "unknown", "error": str(e)}

    response = {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "service": "rag-service",
        "environment": settings.environment,
        "components": components,
    }

    status_code = status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(content=response, status_code=status_code)


@router.get("/health/ready", summary="Readiness probe")
async def readiness_check() -> dict[str, str]:
    """
    Kubernetes-style readiness probe.

    Service is ready if it can process requests.

    Returns:
        Ready status
    """
    try:
        # Check minimum requirements to process requests
        import sys
        from pathlib import Path

        retrieval_path = Path(__file__).parent.parent.parent.parent / "retrieval-service" / "src"
        sys.path.insert(0, str(retrieval_path))

        from retrieval.qdrant_client import get_qdrant_store

        store = get_qdrant_store()
        is_healthy = await store.health_check()

        if is_healthy:
            return {"status": "ready"}
        else:
            return JSONResponse(
                content={"status": "not_ready", "reason": "Qdrant unavailable"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={"status": "not_ready", "reason": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


@router.get("/health/live", summary="Liveness probe")
async def liveness_check() -> dict[str, str]:
    """
    Kubernetes-style liveness probe.

    Service is alive if the process is running.

    Returns:
        Alive status
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> dict[str, Any]:
    """
    Basic metrics endpoint for monitoring.

    Returns:
        Service metrics
    """
    # TODO: Integrate with Prometheus client library
    # For now, return basic stats

    try:
        import sys
        from pathlib import Path

        retrieval_path = Path(__file__).parent.parent.parent.parent / "retrieval-service" / "src"
        sys.path.insert(0, str(retrieval_path))

        from retrieval.qdrant_client import get_qdrant_store

        store = get_qdrant_store()
        info = await store.get_collection_info()

        return {
            "qdrant_documents_total": info.get("points_count", 0),
            "service_uptime_seconds": 0,  # TODO: Track uptime
            "queries_total": 0,  # TODO: Track queries
            "errors_total": 0,  # TODO: Track errors
        }

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e)}
