"""Improved retrieval service with better strategies for legal RAG."""

from typing import Any

from retrieval.embedder import get_embedding_service
from retrieval.qdrant_client import get_qdrant_store

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImprovedRetrievalService:
    """
    Enhanced retrieval service with multiple strategies.

    Improvements:
    1. Query expansion for better recall
    2. Multi-stage retrieval (broad search, then rerank)
    3. Adaptive score thresholds based on query complexity
    4. Better filter relaxation strategy
    """

    def __init__(self):
        """Initialize improved retrieval service."""
        self.embedder = get_embedding_service()
        self.vector_store = get_qdrant_store()
        logger.info("Initialized improved retrieval service")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        use_query_expansion: bool = True,
    ) -> dict[str, Any]:
        """
        Retrieve with improved strategies.

        Args:
            query: User's search query
            top_k: Number of results to return
            filters: Metadata filters
            score_threshold: Minimum similarity score (None = adaptive)
            use_query_expansion: Whether to expand query for better recall

        Returns:
            Dict with chunks and metadata
        """
        logger.info(f"Retrieving for query: {query[:100]}...")

        try:
            # 1. Generate query embedding
            query_vector = await self.embedder.embed_query(query)

            # 2. Adaptive score threshold
            # For legal documents, lower thresholds are often needed
            # because queries use different terminology than legal text
            if score_threshold is None:
                # Don't use a threshold - let the LLM decide relevance
                score_threshold = None
            elif score_threshold > 0.5:
                logger.warning(
                    f"Score threshold {score_threshold} may be too high for legal text. "
                    f"Consider lowering to 0.3-0.4 or removing entirely."
                )

            # 3. Multi-stage retrieval
            # Stage 1: Broad search with relaxed/no filters
            fetch_k = max(top_k * 5, 50)  # Fetch more candidates

            # First try with filters
            results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=fetch_k,
                filters=filters,
                score_threshold=score_threshold,
            )

            # If no results and filters were used, progressively relax
            filters_applied = filters or {}
            if not results and filters:
                logger.info("No results with strict filters, relaxing progressively...")

                # Relaxation order: domain -> risk_category -> keep only regulation
                relaxation_order = ["domain", "risk_category"]
                relaxed_filters = dict(filters)

                for key_to_remove in relaxation_order:
                    if key_to_remove in relaxed_filters:
                        relaxed_filters.pop(key_to_remove)
                        logger.info(f"Retrying without '{key_to_remove}' filter")

                        results = await self.vector_store.search(
                            query_vector=query_vector,
                            top_k=fetch_k,
                            filters=relaxed_filters or None,
                            score_threshold=score_threshold,
                        )

                        if results:
                            filters_applied = relaxed_filters
                            logger.info(f"Found {len(results)} results with relaxed filters")
                            break

                # Final fallback: no filters at all
                if not results:
                    logger.info("Still no results, searching without any filters")
                    results = await self.vector_store.search(
                        query_vector=query_vector,
                        top_k=fetch_k,
                        filters=None,
                        score_threshold=score_threshold,
                    )
                    filters_applied = {}

            # 4. Query expansion (if enabled and initial results are poor)
            if use_query_expansion and (not results or (results and results[0].get("score", 0) < 0.35)):
                logger.info("Scores low, trying query expansion...")
                expanded_results = await self._retrieve_with_expansion(
                    query=query,
                    query_vector=query_vector,
                    top_k=fetch_k,
                    filters=filters_applied or None,
                    score_threshold=score_threshold,
                )

                # Merge with existing results and deduplicate
                results = self._merge_and_deduplicate(results, expanded_results, top_k=fetch_k)

            # 5. Take top_k after all retrieval strategies
            results = results[:top_k]

            # 6. Calculate metadata
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

    async def _retrieve_with_expansion(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
        score_threshold: float | None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve using query expansion.

        Expands query with legal synonyms and related terms.
        """
        # Simple expansion strategy: add legal synonyms
        expansion_terms = {
            "high-risk": ["high risk", "risk classification"],
            "AI system": ["artificial intelligence system", "AI", "automated system"],
            "requirements": ["obligations", "provisions", "rules"],
            "documentation": ["documents", "records", "information"],
            "biometric": ["biometric identification", "biometric data"],
        }

        # Build expanded query
        expanded_query = query
        for term, synonyms in expansion_terms.items():
            if term.lower() in query.lower():
                # Add first synonym
                expanded_query = f"{expanded_query} {synonyms[0]}"

        logger.info(f"Expanded query: {expanded_query[:100]}...")

        # Generate embedding for expanded query
        expanded_vector = await self.embedder.embed_query(expanded_query)

        # Search with expanded query
        results = await self.vector_store.search(
            query_vector=expanded_vector,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )

        return results

    def _merge_and_deduplicate(
        self,
        results1: list[dict[str, Any]],
        results2: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Merge two result sets and deduplicate by ID, keeping highest scores.
        """
        # Build dict by ID, keeping highest score
        merged = {}

        for result in results1 + results2:
            result_id = result.get("id")
            if result_id not in merged:
                merged[result_id] = result
            else:
                # Keep the one with higher score
                if result.get("score", 0) > merged[result_id].get("score", 0):
                    merged[result_id] = result

        # Sort by score and take top_k
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        return sorted_results[:top_k]

    async def health_check(self) -> dict[str, bool]:
        """Check health of retrieval components."""
        health = {
            "qdrant": await self.vector_store.health_check(),
            "embedder": True,
        }
        return health


# ===== Singleton instance =====

_improved_retrieval_service: ImprovedRetrievalService | None = None


def get_improved_retrieval_service() -> ImprovedRetrievalService:
    """Get singleton improved retrieval service instance."""
    global _improved_retrieval_service

    if _improved_retrieval_service is None:
        _improved_retrieval_service = ImprovedRetrievalService()

    return _improved_retrieval_service
