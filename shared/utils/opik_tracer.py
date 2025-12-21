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
_opik_client = None
_current_trace = None


def get_opik_client():
    """
    Get or create Opik client.

    Returns:
        Opik client instance or None if Opik is disabled
    """
    global _opik_client

    if not settings.opik_enabled:
        return None

    if _opik_client is None:
        try:
            import opik

            _opik_client = opik.Opik(
                api_key=settings.opik_api_key,
                workspace=settings.opik_workspace,
                project=settings.opik_project,
                url=settings.opik_url,
            )
            logger.info(
                f"✓ Opik client initialized "
                f"(workspace: {settings.opik_workspace}, project: {settings.opik_project})"
            )
        except ImportError:
            logger.warning("Opik package not installed. Install with: pip install opik")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Opik client: {str(e)}")
            return None

    return _opik_client


@contextmanager
def trace_context(name: str, tags: list[str] | None = None, metadata: dict | None = None):
    """
    Context manager for creating a trace span.

    Args:
        name: Name of the trace span
        tags: Optional list of tags
        metadata: Optional metadata dict

    Yields:
        Trace span object
    """
    client = get_opik_client()

    if client is None:
        yield None
        return

    try:
        import opik

        with opik.track(
            name=name,
            tags=tags or ["conformai"],
            metadata=metadata or {},
        ) as trace:
            yield trace

    except Exception as e:
        logger.warning(f"Opik trace failed: {e}")
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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                # Opik disabled, just run the function
                return func(*args, **kwargs)

            try:
                import opik

                # Start trace
                with opik.track(
                    name=operation_name,
                    tags=[operation_type, "conformai"],
                    metadata=metadata or {},
                ) as trace:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Log success
                    trace.update(output={"status": "success"})

                    return result

            except Exception as e:
                # Log error to Opik
                if client:
                    try:
                        trace.update(
                            output={"status": "error", "error": str(e)}, error=True
                        )
                    except:
                        pass

                # Re-raise the exception
                raise

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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                return func(*args, **kwargs)

            try:
                import opik

                # Extract prompt from args/kwargs
                prompt = kwargs.get("prompt") or (args[0] if args else "")

                with opik.track(
                    name=f"llm_call_{model_name}",
                    tags=["llm", provider, "conformai"],
                    metadata={
                        "model": model_name,
                        "provider": provider,
                        "temperature": temperature,
                    },
                ) as trace:
                    # Log input
                    trace.log_input({"prompt": str(prompt)[:1000]})  # First 1000 chars

                    # Execute LLM call
                    result = func(*args, **kwargs)

                    # Log output
                    trace.log_output({"response": str(result)[:1000]})

                    return result

            except Exception as e:
                if client:
                    try:
                        trace.update(output={"status": "error", "error": str(e)}, error=True)
                    except:
                        pass
                raise

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
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                return func(*args, **kwargs)

            try:
                import opik

                with opik.track(
                    name="generate_embeddings",
                    tags=["embedding", "openai", "conformai"],
                    metadata={"model": model_name},
                ) as trace:
                    # Count texts
                    texts = args[0] if args else kwargs.get("texts", [])
                    num_texts = len(texts) if isinstance(texts, list) else 1

                    trace.log_input({"num_texts": num_texts})

                    # Execute embedding generation
                    result = func(*args, **kwargs)

                    # Log output
                    trace.log_output(
                        {
                            "num_embeddings": len(result)
                            if isinstance(result, list)
                            else 1,
                            "status": "success",
                        }
                    )

                    return result

            except Exception as e:
                if client:
                    try:
                        trace.update(output={"status": "error", "error": str(e)}, error=True)
                    except:
                        pass
                raise

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
    client = get_opik_client()

    if client is None:
        return

    try:
        client.log_metric(
            name=metric_name,
            value=value,
            tags=tags or {},
        )
    except Exception as e:
        logger.warning(f"Failed to log metric to Opik: {str(e)}")


def log_event(event_name: str, properties: dict[str, Any] | None = None):
    """
    Log a custom event to Opik.

    Args:
        event_name: Name of the event
        properties: Event properties
    """
    client = get_opik_client()

    if client is None:
        return

    try:
        client.log_event(
            name=event_name,
            properties=properties or {},
        )
    except Exception as e:
        logger.warning(f"Failed to log event to Opik: {str(e)}")


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
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                return await func(*args, **kwargs)

            start_time = time.time()

            try:
                import opik

                # Extract state from args
                state = args[0] if args else kwargs.get("state", {})

                # Prepare metadata
                metadata = {
                    "node_name": node_name,
                    "node_type": node_type,
                    "query": state.get("query", "")[:200] if isinstance(state, dict) else "",
                    "iteration": state.get("iteration_count", 0) if isinstance(state, dict) else 0,
                }

                with opik.track(
                    name=f"langgraph_node_{node_name}",
                    tags=["langgraph", node_type, "conformai"],
                    metadata=metadata,
                ) as trace:
                    # Execute node
                    result = await func(*args, **kwargs)

                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000

                    # Log output
                    output_data = {
                        "status": "success",
                        "duration_ms": duration_ms,
                    }

                    # Add specific output based on node type
                    if node_type == "analysis" and isinstance(result, dict):
                        output_data.update({
                            "intent": result.get("intent"),
                            "complexity": result.get("query_complexity"),
                            "ai_domain": str(result.get("ai_domain")) if result.get("ai_domain") else None,
                        })
                    elif node_type == "retrieval" and isinstance(result, dict):
                        output_data.update({
                            "chunks_retrieved": len(result.get("all_retrieved_chunks", [])),
                        })
                    elif node_type == "synthesis" and isinstance(result, dict):
                        output_data.update({
                            "answer_length": len(result.get("final_answer", "")),
                            "citation_count": len(result.get("citations", [])),
                        })

                    trace.update(output=output_data)

                    return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if client:
                    try:
                        trace.update(
                            output={
                                "status": "error",
                                "error": str(e),
                                "duration_ms": duration_ms,
                            },
                            error=True,
                        )
                    except:
                        pass
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                return func(*args, **kwargs)

            start_time = time.time()

            try:
                import opik

                state = args[0] if args else kwargs.get("state", {})

                metadata = {
                    "node_name": node_name,
                    "node_type": node_type,
                    "query": state.get("query", "")[:200] if isinstance(state, dict) else "",
                    "iteration": state.get("iteration_count", 0) if isinstance(state, dict) else 0,
                }

                with opik.track(
                    name=f"langgraph_node_{node_name}",
                    tags=["langgraph", node_type, "conformai"],
                    metadata=metadata,
                ) as trace:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    trace.update(output={
                        "status": "success",
                        "duration_ms": duration_ms,
                    })

                    return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if client:
                    try:
                        trace.update(
                            output={
                                "status": "error",
                                "error": str(e),
                                "duration_ms": duration_ms,
                            },
                            error=True,
                        )
                    except:
                        pass
                raise

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
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            client = get_opik_client()

            if client is None:
                return await func(*args, **kwargs)

            start_time = time.time()

            try:
                import opik

                # Extract query from args
                query = args[0] if args else kwargs.get("query", "")

                with opik.track(
                    name=pipeline_name,
                    tags=["rag_pipeline", "end_to_end", "conformai"],
                    metadata={
                        "query": query[:200] if isinstance(query, str) else "",
                        "pipeline": pipeline_name,
                    },
                ) as trace:
                    # Log input
                    trace.log_input({"query": query})

                    # Execute pipeline
                    result = await func(*args, **kwargs)

                    # Calculate duration
                    duration_ms = (time.time() - start_time) * 1000

                    # Extract metrics from result
                    output_data = {
                        "status": "success",
                        "duration_ms": duration_ms,
                    }

                    if isinstance(result, dict):
                        output_data.update({
                            "answer_length": len(result.get("final_answer", "")),
                            "citation_count": len(result.get("citations", [])),
                            "confidence_score": result.get("confidence_score", 0.0),
                            "iterations": result.get("iteration_count", 0),
                            "llm_calls": result.get("total_llm_calls", 0),
                            "tokens_used": result.get("total_tokens_used", 0),
                            "refused": bool(result.get("refusal_reason")),
                        })

                        # Log answer
                        trace.log_output({
                            "answer": result.get("final_answer", "")[:500],
                            "metrics": output_data,
                        })

                    trace.update(output=output_data)

                    # Log performance metrics
                    log_metric("rag_pipeline_duration_ms", duration_ms, {"pipeline": pipeline_name})
                    if isinstance(result, dict):
                        log_metric("rag_confidence_score", result.get("confidence_score", 0.0))
                        log_metric("rag_iterations", result.get("iteration_count", 0))

                    return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                if client:
                    try:
                        trace.update(
                            output={
                                "status": "error",
                                "error": str(e),
                                "duration_ms": duration_ms,
                            },
                            error=True,
                        )
                    except:
                        pass
                raise

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
