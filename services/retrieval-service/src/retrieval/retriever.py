"""Main retrieval service orchestrating embedding, search, and reranking."""

from typing import Any

from retrieval.embedder import get_embedding_service
from retrieval.qdrant_client import get_qdrant_store

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class RetrievalService:
    """
    High-level retrieval service.

    Orchestrates:
    1. Query embedding generation
    2. Vector search with metadata filtering
    3. Optional reranking
    4. Result formatting
    """

    def __init__(self):
        """Initialize retrieval service."""
        self.embedder = get_embedding_service()
        self.vector_store = get_qdrant_store()

        logger.info("Initialized retrieval service")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant legal chunks for a query.

        Args:
            query: User's search query
            top_k: Number of results to return
            filters: Metadata filters (regulation, domain, risk_category, etc.)
            score_threshold: Minimum similarity score (0.0-1.0)
            rerank: Whether to apply reranking (default: False)

        Returns:
            Dict with chunks and metadata

        Example:
            result = await retriever.retrieve(
                query="What are high-risk AI obligations?",
                top_k=10,
                filters={"regulation": "EU AI Act", "risk_category": "high"}
            )
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        try:
            # 1. Generate query embedding
            query_vector = await self.embedder.embed_query(query)

            # 2. Search vector database
            # Fetch more if reranking (fetch_k)
            fetch_k = settings.retrieval_fetch_k if rerank else top_k

            results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=fetch_k,
                filters=filters,
                score_threshold=score_threshold,
            )

            filters_applied = filters or {}
            if not results and filters:
                relaxed_filters = dict(filters)
                for key in ["risk_category", "domain", "regulation"]:
                    if key in relaxed_filters:
                        relaxed_filters.pop(key)
                        logger.info(
                            f"No results with filters; retrying without {key} filter"
                        )
                        results = await self.vector_store.search(
                            query_vector=query_vector,
                            top_k=fetch_k,
                            filters=relaxed_filters or None,
                            score_threshold=score_threshold,
                        )
                        if results:
                            filters_applied = relaxed_filters
                            break

            # 3. Apply reranking if requested
            if rerank and len(results) > top_k:
                # TODO: Implement cross-encoder reranking
                # For now, just take top_k
                results = results[:top_k]
                logger.info("Reranking requested but not yet implemented, using top results")

            # 4. Calculate metadata
            scores = [r.get("score", 0.0) for r in results]

            return {
                "query": query,
                "chunks": results,
                "count": len(results),
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "filters_applied": filters_applied,
            }

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise

    async def retrieve_by_article(
        self,
        regulation: str,
        article: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve a specific article by regulation name and article number.

        Args:
            regulation: Regulation name (e.g., "EU AI Act", "GDPR")
            article: Article number (e.g., "Article 9", "Article 22")

        Returns:
            Article chunk dict or None if not found

        Example:
            chunk = await retriever.retrieve_by_article("EU AI Act", "Article 9")
        """
        logger.info(f"Retrieving {article} from {regulation}")

        try:
            # Search with exact metadata match
            query = f"{regulation} {article}"
            query_vector = await self.embedder.embed_query(query)

            results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=1,
                filters={
                    "regulation": regulation,
                    "article": article,
                },
            )

            if results:
                logger.info(f"Found {article} in {regulation}")
                return results[0]
            else:
                logger.warning(f"Article not found: {article} in {regulation}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving article: {e}")
            return None

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        semantic_weight: float = 0.7,
    ) -> dict[str, Any]:
        """
        Perform hybrid search combining semantic and keyword matching.

        Args:
            query: User's search query
            top_k: Number of results to return
            filters: Metadata filters
            semantic_weight: Weight for semantic search (0.0-1.0), keyword is (1 - semantic_weight)

        Returns:
            Dict with chunks and metadata

        Note:
            Currently uses semantic-only. Full hybrid with BM25 requires
            additional indexing in Qdrant or external keyword search.
        """
        # TODO: Implement true hybrid search with BM25
        # For now, fall back to semantic search
        logger.info("Hybrid search requested, using semantic search for now")

        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    async def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve for multiple queries in batch.

        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Metadata filters applied to all queries

        Returns:
            List of result dicts, one per query
        """
        logger.info(f"Batch retrieving for {len(queries)} queries")

        try:
            # 1. Generate embeddings in batch
            query_vectors = await self.embedder.embed_batch(queries)

            # 2. Batch search
            batch_results = await self.vector_store.batch_search(
                query_vectors=query_vectors,
                top_k=top_k,
                filters=filters,
            )

            # 3. Format results
            results = []
            for query, chunks in zip(queries, batch_results):
                scores = [c.get("score", 0.0) for c in chunks]

                results.append({
                    "query": query,
                    "chunks": chunks,
                    "count": len(chunks),
                    "min_score": min(scores) if scores else 0.0,
                    "max_score": max(scores) if scores else 0.0,
                    "avg_score": sum(scores) / len(scores) if scores else 0.0,
                    "filters_applied": filters or {},
                })

            logger.info(f"Batch retrieval completed for {len(queries)} queries")

            return results

        except Exception as e:
            logger.error(f"Error during batch retrieval: {e}")
            raise

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all retrieval components.

        Returns:
            Dict with health status of each component
        """
        health = {
            "qdrant": await self.vector_store.health_check(),
            "embedder": True,  # If initialized, it's healthy
        }

        return health


# ===== Singleton instance =====

_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    """
    Get singleton retrieval service instance.

    Returns:
        RetrievalService instance
    """
    global _retrieval_service

    if _retrieval_service is None:
        _retrieval_service = RetrievalService()

    return _retrieval_service
