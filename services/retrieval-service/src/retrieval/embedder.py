"""Query embedding generation."""

from functools import lru_cache

from openai import AsyncOpenAI

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating query embeddings using OpenAI."""

    def __init__(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimension

        logger.info(
            f"Initialized embedding service: model={self.model}, "
            f"dimensions={self.dimensions}"
        )

    async def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding vector for a single query.

        Args:
            query: Text query to embed

        Returns:
            Embedding vector as list of floats

        Example:
            vector = await embedder.embed_query("What are high-risk AI obligations?")
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=query,
                dimensions=self.dimensions,
            )

            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding for query: {query[:50]}...")

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_batch(self, queries: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple queries in batch.

        Args:
            queries: List of text queries

        Returns:
            List of embedding vectors

        Example:
            vectors = await embedder.embed_batch([
                "What are obligations?",
                "What are prohibitions?"
            ])
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=queries,
                dimensions=self.dimensions,
            )

            embeddings = [item.embedding for item in response.data]

            logger.info(f"Generated {len(embeddings)} embeddings in batch")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise


# ===== Singleton instance =====

_embedding_service: EmbeddingService | None = None


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance.

    Returns:
        EmbeddingService instance
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService()

    return _embedding_service
