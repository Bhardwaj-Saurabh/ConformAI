"""
Opik Observability Integration

Provides tracing and monitoring for ConformAI using Opik.
"""

import functools
from typing import Any, Callable, Optional

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global Opik client (lazy loaded)
_opik_client = None


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


def track_operation(
    operation_name: str,
    operation_type: str = "general",
    metadata: Optional[dict[str, Any]] = None,
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


def log_metric(metric_name: str, value: float, tags: Optional[dict[str, str]] = None):
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


def log_event(event_name: str, properties: Optional[dict[str, Any]] = None):
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
