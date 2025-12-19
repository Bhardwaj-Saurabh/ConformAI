"""LLM client factory for Anthropic and OpenAI models."""

from functools import lru_cache
import asyncio
from typing import Literal

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

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
    """
    try:
        return await llm.ainvoke(input_data)
    except Exception as exc:
        logger.warning(f"Async LLM call failed, falling back to sync invoke: {exc}")
        return await asyncio.to_thread(llm.invoke, input_data)
