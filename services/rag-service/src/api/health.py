"""
Health check endpoints for RAG service.

Production-ready health monitoring with comprehensive checks for:
- Vector database (Qdrant)
- LLM API availability
- System resources (CPU, memory, disk)
- Service dependencies
"""

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
import psutil

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

router = APIRouter(tags=["Health"])
settings = get_settings()
logger = get_logger(__name__)

# Prometheus metrics
health_check_counter = Counter(
    "rag_service_health_checks_total", "Total health check requests", ["status"]
)
qdrant_documents_gauge = Gauge("rag_service_qdrant_documents", "Number of documents in Qdrant")
system_cpu_gauge = Gauge("rag_service_cpu_percent", "CPU usage percentage")
system_memory_gauge = Gauge("rag_service_memory_percent", "Memory usage percentage")
query_counter = Counter("rag_service_queries_total", "Total queries processed")
query_latency = Histogram(
    "rag_service_query_latency_seconds", "Query processing latency", buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)


async def _check_qdrant() -> dict[str, Any]:
    """Check Qdrant vector database health."""
    try:
        import sys
        from pathlib import Path

        retrieval_path = Path(__file__).parent.parent.parent.parent / "retrieval-service" / "src"
        sys.path.insert(0, str(retrieval_path))

        from retrieval.qdrant_client import get_qdrant_store

        store = get_qdrant_store()
        qdrant_healthy = await store.health_check()

        if qdrant_healthy:
            info = await store.get_collection_info()
            points_count = info.get("points_count", 0)

            # Update Prometheus gauge
            qdrant_documents_gauge.set(points_count)

            return {
                "status": "up",
                "collection": info.get("name"),
                "documents_indexed": points_count,
                "vectors_count": info.get("vectors_count", 0),
            }
        else:
            return {"status": "down", "error": "Connection failed"}

    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}", exc_info=True)
        return {"status": "down", "error": str(e)}


async def _check_llm_api() -> dict[str, Any]:
    """Check LLM API health."""
    try:
        # Check if API key is configured
        if settings.llm_provider == "anthropic":
            api_key = settings.anthropic_api_key
        elif settings.llm_provider == "openai":
            api_key = settings.openai_api_key
        else:
            return {"status": "unknown", "error": "Unknown LLM provider"}

        if not api_key or api_key == "your-api-key-here":
            return {
                "status": "degraded",
                "provider": settings.llm_provider,
                "model": settings.llm_model,
                "error": "API key not configured",
            }

        return {
            "status": "up",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "configured": True,
        }

    except Exception as e:
        logger.error(f"LLM API health check failed: {e}", exc_info=True)
        return {"status": "down", "error": str(e)}


async def _get_system_metrics() -> dict[str, Any]:
    """Get system resource metrics."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Update Prometheus gauges
        system_cpu_gauge.set(cpu)
        system_memory_gauge.set(memory.percent)

        is_healthy = cpu < 90 and memory.percent < 90 and disk.percent < 90

        return {
            "status": "healthy" if is_healthy else "degraded",
            "cpu_percent": round(cpu, 1),
            "memory_percent": round(memory.percent, 1),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": round(disk.percent, 1),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "healthy": is_healthy,
        }

    except Exception as e:
        logger.error(f"System metrics check failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "healthy": False}


@router.get("/health", summary="Comprehensive health check")
async def health_check() -> dict[str, Any]:
    """
    Comprehensive health check for RAG service.

    Checks:
    - Qdrant connectivity and document count
    - LLM API configuration
    - System resources (CPU, memory, disk)

    Returns:
        Health status with component details
    """
    logger.info("Health check requested")

    # Run all checks in parallel for faster response
    qdrant_check, llm_check, system_check = await asyncio.gather(
        _check_qdrant(), _check_llm_api(), _get_system_metrics(), return_exceptions=True
    )

    # Handle exceptions in checks
    if isinstance(qdrant_check, Exception):
        qdrant_check = {"status": "down", "error": str(qdrant_check)}
    if isinstance(llm_check, Exception):
        llm_check = {"status": "down", "error": str(llm_check)}
    if isinstance(system_check, Exception):
        system_check = {"status": "error", "error": str(system_check), "healthy": False}

    # Determine overall health
    qdrant_healthy = qdrant_check.get("status") == "up"
    llm_healthy = llm_check.get("status") in ["up", "degraded"]
    system_healthy = system_check.get("healthy", False)

    if qdrant_healthy and llm_healthy and system_healthy:
        overall_status = "healthy"
    elif qdrant_check.get("status") == "down" or not system_healthy:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    # Update Prometheus counter
    health_check_counter.labels(status=overall_status).inc()

    response = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "rag-service",
        "environment": settings.environment,
        "version": "0.1.0",  # TODO: Get from package version
        "components": {"qdrant": qdrant_check, "llm_api": llm_check, "system": system_check},
    }

    # Log unhealthy status
    if overall_status != "healthy":
        logger.warning(f"Service health is {overall_status}", extra={"health_response": response})

    status_code = status.HTTP_200_OK if overall_status == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE

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
async def metrics() -> PlainTextResponse:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping by Prometheus server.

    Metrics include:
    - rag_service_health_checks_total: Total health check requests by status
    - rag_service_qdrant_documents: Number of documents in Qdrant
    - rag_service_cpu_percent: CPU usage percentage
    - rag_service_memory_percent: Memory usage percentage
    - rag_service_queries_total: Total queries processed
    - rag_service_query_latency_seconds: Query processing latency histogram

    Returns:
        Prometheus-formatted metrics
    """
    logger.debug("Metrics requested")

    # Update gauges with latest values
    try:
        qdrant_check = await _check_qdrant()
        system_check = await _get_system_metrics()
    except Exception as e:
        logger.error(f"Failed to update metrics: {e}", exc_info=True)

    # Generate Prometheus format
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health/startup", summary="Startup probe")
async def startup_check() -> dict[str, Any]:
    """
    Kubernetes-style startup probe.

    Checks if the service has completed initialization.
    Used for slow-starting services.

    Returns:
        Startup status
    """
    logger.debug("Startup check requested")

    try:
        # Check if Qdrant collection is initialized
        qdrant_check = await _check_qdrant()
        qdrant_initialized = qdrant_check.get("status") == "up" and qdrant_check.get("documents_indexed", 0) > 0

        if not qdrant_initialized:
            logger.warning("Service not fully started - Qdrant not initialized")
            return JSONResponse(
                content={
                    "status": "starting",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "qdrant_initialized": qdrant_initialized,
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        return {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "initialized": {"qdrant": qdrant_initialized},
        }

    except Exception as e:
        logger.error(f"Startup check failed: {e}", exc_info=True)
        return JSONResponse(
            content={"status": "starting", "error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
