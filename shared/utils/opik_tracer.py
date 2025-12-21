"""
Opik Observability Integration

Provides tracing and monitoring for ConformAI using Opik.
Includes LangGraph node tracing, LLM call tracking, and performance metrics.
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global Opik client (lazy loaded)
_opik_configured = False


def configure_opik():
    """
    Configure Opik for the project.

    Returns:
        bool: True if configured successfully, False otherwise
    """
    global _opik_configured

    if not settings.opik_enabled:
        return False

    if _opik_configured:
        return True

    try:
        import opik

        # Configure Opik
        # Note: The opik.configure() API varies by version
        # Use API key only - workspace is determined by the key
        opik.configure(
            api_key=settings.opik_api_key,
            use_local=False,
        )

        _opik_configured = True
        logger.info(
            f"✓ Opik configured "
            f"(workspace: {settings.opik_workspace}, project: {settings.opik_project})"
        )
        return True

    except ImportError:
        logger.warning("Opik package not installed. Install with: pip install opik")
        return False
    except Exception as e:
        logger.error(f"Failed to configure Opik: {str(e)}")
        return False


def get_opik_client():
    """
    Get Opik client (for backward compatibility).
    Now just ensures Opik is configured.

    Returns:
        bool: True if Opik is configured, None otherwise
    """
    return configure_opik() or None


@contextmanager
def trace_context(name: str, tags: list[str] | None = None, metadata: dict | None = None):
    """
    Context manager for creating a trace span (stub for backward compatibility).

    Args:
        name: Name of the trace span
        tags: Optional list of tags
        metadata: Optional metadata dict

    Yields:
        None (no-op for now)
    """
    # Simple no-op context manager for backward compatibility
    # Opik tracing is now done via decorators
    yield None


def track_operation(
    operation_name: str,
    operation_type: str = "general",
    metadata: dict[str, Any] | None = None,
):
    """
    Decorator to track function execution with Opik.

    Args:
        operation_name: Name of the operation (e.g., "embed_chunks", "parse_document")
        operation_type: Type of operation (e.g., "embedding", "parsing", "retrieval")
        metadata: Additional metadata to log

    Example:
        @track_operation("generate_embeddings", "embedding")
        def generate_embeddings(chunks):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not settings.opik_enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not configure_opik():
                return func(*args, **kwargs)

            try:
                import opik

                @opik.track(
                    name=operation_name,
                    tags=[operation_type, "conformai"],
                    metadata=metadata or {},
                )
                def tracked_func():
                    return func(*args, **kwargs)

                return tracked_func()

            except Exception as e:
                logger.debug(f"Opik tracking failed for {operation_name}: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_llm_call(
    model_name: str,
    provider: str = "anthropic",
    temperature: float = 0.0,
):
    """
    Decorator to track LLM calls with Opik.

    Args:
        model_name: Name of the LLM model
        provider: LLM provider (e.g., "anthropic", "openai")
        temperature: LLM temperature

    Example:
        @track_llm_call("claude-3-5-sonnet-20241022", "anthropic")
        def generate_answer(prompt):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not settings.opik_enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not configure_opik():
                return func(*args, **kwargs)

            try:
                import opik

                @opik.track(
                    name=f"llm_call_{model_name}",
                    tags=["llm", provider, "conformai"],
                    metadata={
                        "model": model_name,
                        "provider": provider,
                        "temperature": temperature,
                    },
                )
                def tracked_func():
                    return func(*args, **kwargs)

                return tracked_func()

            except Exception as e:
                logger.debug(f"Opik LLM tracking failed: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def track_embedding_call(model_name: str = "text-embedding-3-large"):
    """
    Decorator to track embedding generation with Opik.

    Args:
        model_name: Name of embedding model

    Example:
        @track_embedding_call("text-embedding-3-large")
        def generate_embeddings(texts):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not settings.opik_enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not configure_opik():
                return func(*args, **kwargs)

            try:
                import opik

                @opik.track(
                    name="generate_embeddings",
                    tags=["embedding", "openai", "conformai"],
                    metadata={"model": model_name},
                )
                def tracked_func():
                    return func(*args, **kwargs)

                return tracked_func()

            except Exception as e:
                logger.debug(f"Opik embedding tracking failed: {e}")
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_metric(metric_name: str, value: float, tags: dict[str, str] | None = None):
    """
    Log a custom metric to Opik.

    Args:
        metric_name: Name of the metric
        value: Metric value
        tags: Optional tags for the metric
    """
    if not settings.opik_enabled:
        return

    # Opik SDK doesn't expose direct metric logging in newer versions
    # Metrics are automatically collected from tracked functions
    logger.debug(f"Metric logged: {metric_name}={value} (tags: {tags})")


def log_event(event_name: str, properties: dict[str, Any] | None = None):
    """
    Log a custom event to Opik.

    Args:
        event_name: Name of the event
        properties: Event properties
    """
    if not settings.opik_enabled:
        return

    # Opik SDK doesn't expose direct event logging in newer versions
    # Events are automatically collected from tracked functions
    logger.debug(f"Event logged: {event_name} (properties: {properties})")


def track_langgraph_node(node_name: str, node_type: str = "processing"):
    """
    Decorator to track LangGraph node execution with Opik.

    Args:
        node_name: Name of the LangGraph node
        node_type: Type of node (analysis, retrieval, synthesis, validation, etc.)

    Example:
        @track_langgraph_node("analyze_query", "analysis")
        async def analyze_query(state: RAGState):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not settings.opik_enabled:
            return func

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not configure_opik():
                return await func(*args, **kwargs)

            try:
                import opik

                # Extract state from args
                state = args[0] if args else kwargs.get("state", {})

                # Prepare metadata
                metadata = {
                    "node_name": node_name,
                    "node_type": node_type,
                    "query": state.get("query", "")[:200] if isinstance(state, dict) else "",
                    "iteration": state.get("iteration_count", 0)
                    if isinstance(state, dict)
                    else 0,
                }

                @opik.track(
                    name=f"langgraph_{node_name}",
                    tags=["langgraph", node_type, "conformai"],
                    metadata=metadata,
                )
                async def tracked_func():
                    return await func(*args, **kwargs)

                return await tracked_func()

            except Exception as e:
                logger.debug(f"Opik LangGraph tracking failed for {node_name}: {e}")
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not configure_opik():
                return func(*args, **kwargs)

            try:
                import opik

                state = args[0] if args else kwargs.get("state", {})

                metadata = {
                    "node_name": node_name,
                    "node_type": node_type,
                    "query": state.get("query", "")[:200] if isinstance(state, dict) else "",
                    "iteration": state.get("iteration_count", 0)
                    if isinstance(state, dict)
                    else 0,
                }

                @opik.track(
                    name=f"langgraph_{node_name}",
                    tags=["langgraph", node_type, "conformai"],
                    metadata=metadata,
                )
                def tracked_func():
                    return func(*args, **kwargs)

                return tracked_func()

            except Exception as e:
                logger.debug(f"Opik LangGraph tracking failed for {node_name}: {e}")
                return func(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_rag_pipeline(pipeline_name: str = "rag_pipeline"):
    """
    Decorator to track the entire RAG pipeline execution.

    Args:
        pipeline_name: Name of the pipeline

    Example:
        @track_rag_pipeline("eu_compliance_rag")
        async def run_rag_pipeline(query: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        if not settings.opik_enabled:
            return func

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not configure_opik():
                return await func(*args, **kwargs)

            try:
                import opik

                # Extract query from args
                query = args[0] if args else kwargs.get("query", "")

                @opik.track(
                    name=pipeline_name,
                    tags=["rag_pipeline", "end_to_end", "conformai"],
                    metadata={
                        "query": query[:200] if isinstance(query, str) else "",
                        "pipeline": pipeline_name,
                    },
                )
                async def tracked_func():
                    return await func(*args, **kwargs)

                return await tracked_func()

            except Exception as e:
                logger.debug(f"Opik RAG pipeline tracking failed: {e}")
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Test Opik integration
    @track_operation("test_operation", "test")
    def test_function(x: int, y: int) -> int:
        return x + y

    result = test_function(2, 3)
    print(f"Result: {result}")

    # Log test metric
    log_metric("test_metric", 42.0, {"environment": "development"})

    # Log test event
    log_event("test_event", {"message": "Hello from ConformAI"})
