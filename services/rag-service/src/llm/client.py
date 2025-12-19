"""LLM client factory for Anthropic and OpenAI models."""

from functools import lru_cache
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

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

        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

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
        # Use Sonnet for planning
        return get_llm_client(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.0,
        )
    else:
        # Use GPT-4o for planning
        return get_llm_client(
            provider="openai",
            model="gpt-4o",
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
