"""LLM client factory for Anthropic and OpenAI models."""

import asyncio
import time
from functools import lru_cache
from typing import Literal

from shared.config.settings import get_settings
from shared.utils.logger import get_logger
from shared.utils.opik_tracer import log_metric, trace_context

settings = get_settings()
logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_llm_client(
    provider: Literal["anthropic", "openai"] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Get LLM client with caching.

    Args:
        provider: LLM provider ("anthropic" or "openai"). Uses settings default if None.
        model: Model name. Uses settings default if None.
        temperature: Temperature for generation. Uses settings default if None.
        max_tokens: Max tokens to generate. Uses settings default if None.

    Returns:
        LangChain ChatModel instance (ChatAnthropic or ChatOpenAI)

    Example:
        llm = get_llm_client()  # Uses settings defaults
        llm = get_llm_client(provider="anthropic", model="claude-3-5-sonnet-20241022")
    """
    # Use settings defaults if not provided
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens

    logger.info(f"Initializing LLM client: {provider}/{model}")

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for Anthropic models. "
                "Install with: pip install langchain-anthropic"
            ) from exc

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenAI models. "
                "Install with: pip install langchain-openai"
            ) from exc

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_planning_llm():
    """
    Get LLM optimized for agent planning (reasoning-heavy tasks).

    Uses more capable model for complex reasoning tasks like
    query decomposition and ReAct planning.

    Returns:
        LangChain ChatModel for planning tasks
    """
    if settings.llm_provider == "anthropic":
        return get_llm_client(
            provider="anthropic",
            model=settings.llm_model,
            temperature=0.0,
        )

    return get_llm_client(
        provider="openai",
        model=settings.llm_model,
        temperature=0.0,
    )


def get_generation_llm():
    """
    Get LLM optimized for answer generation.

    Can use faster/cheaper model for straightforward generation tasks.

    Returns:
        LangChain ChatModel for generation tasks
    """
    # Use configured default model
    return get_llm_client()


async def invoke_llm(llm, input_data):
    """
    Invoke LLM with async-first strategy and sync fallback.

    Some environments have broken async DNS; fallback uses sync invoke in a thread.
    Includes Opik tracing for observability.
    """
    start_time = time.time()

    # Extract model info for tracing
    model_name = getattr(llm, "model_name", "unknown")
    provider = "anthropic" if "claude" in model_name.lower() else "openai"

    # Prepare input text for logging
    if isinstance(input_data, list):
        input_text = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in input_data])
    else:
        input_text = str(input_data)

    with trace_context(
        name=f"llm_call_{model_name}",
        tags=["llm", provider, "conformai"],
        metadata={
            "model": model_name,
            "provider": provider,
            "input_length": len(input_text),
        }
    ) as trace:
        try:
            # Log input
            if trace:
                trace.log_input({"prompt": input_text[:1000]})  # First 1000 chars

            # Execute LLM call
            response = await llm.ainvoke(input_data)

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            output_text = response.content if hasattr(response, 'content') else str(response)

            # Extract token usage if available
            tokens_used = 0
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)

            # Log output
            if trace:
                trace.log_output({
                    "response": output_text[:1000],  # First 1000 chars
                    "tokens_used": tokens_used,
                })
                trace.update(output={
                    "status": "success",
                    "duration_ms": duration_ms,
                    "tokens_used": tokens_used,
                    "output_length": len(output_text),
                })

            # Log metrics to Opik
            log_metric("llm_call_duration_ms", duration_ms, {"model": model_name, "provider": provider})
            if tokens_used > 0:
                log_metric("llm_tokens_used", tokens_used, {"model": model_name, "provider": provider})

            # Log to production logger
            logger.log_llm_call(
                model=model_name,
                provider=provider,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
                input_length=len(input_text),
                output_length=len(output_text),
            )

            return response

        except Exception as exc:
            # Try sync fallback
            logger.warning(f"Async LLM call failed, falling back to sync invoke: {exc}")

            try:
                response = await asyncio.to_thread(llm.invoke, input_data)

                duration_ms = (time.time() - start_time) * 1000
                output_text = response.content if hasattr(response, 'content') else str(response)

                if trace:
                    trace.log_output({"response": output_text[:1000]})
                    trace.update(output={
                        "status": "success_fallback",
                        "duration_ms": duration_ms,
                        "fallback": "sync",
                    })

                log_metric("llm_call_duration_ms", duration_ms, {"model": model_name, "provider": provider, "fallback": "true"})

                # Log fallback success
                logger.warning(
                    "LLM call succeeded with sync fallback after async failure",
                    extra={
                        "llm_provider": provider,
                        "llm_model": model_name,
                        "duration_ms": duration_ms,
                        "fallback": True,
                    }
                )

                return response

            except Exception as final_exc:
                # Both attempts failed
                duration_ms = (time.time() - start_time) * 1000

                if trace:
                    trace.update(output={
                        "status": "error",
                        "error": str(final_exc),
                        "duration_ms": duration_ms,
                    }, error=True)

                # Log LLM failure
                logger.log_error_with_context(
                    message="LLM call failed completely (async and sync)",
                    error=final_exc,
                    llm_provider=provider,
                    llm_model=model_name,
                    duration_ms=duration_ms,
                    input_length=len(input_text),
                )

                raise
