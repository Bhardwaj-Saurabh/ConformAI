"""Qdrant vector database client wrapper."""

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from shared.config.settings import get_settings
from shared.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Wrapper for Qdrant vector database operations.

    Handles connection management, search, and filtering.
    """

    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        )
        self.collection_name = settings.qdrant_collection_name

        logger.info(f"Initialized Qdrant client for collection: {self.collection_name}")

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search in Qdrant.

        Args:
            query_vector: Embedding vector for the query
            top_k: Number of results to return
            filters: Metadata filters (domain, regulation, risk_category, etc.)
            score_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of search results with content, metadata, and scores

        Example:
            results = await store.search(
                query_vector=[0.1, 0.2, ...],
                top_k=10,
                filters={"regulation": "EU AI Act", "risk_category": "high"}
            )
        """
        try:
            # Build Qdrant filter from dict
            qdrant_filter = self._build_filter(filters) if filters else None

            # Perform search
            if hasattr(self.client, "search"):
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,  # Don't return vectors to save bandwidth
                )
            else:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                )
                search_result = response.points

            # Convert to dict format
            results = [self._point_to_dict(point) for point in search_result]

            logger.info(
                f"Search returned {len(results)} results (top_k={top_k}, "
                f"filters={filters is not None})"
            )

            return results

        except Exception as e:
            logger.error(f"Error during Qdrant search: {e}")
            raise

    async def search_by_id(self, point_id: str) -> dict[str, Any] | None:
        """
        Retrieve a specific document chunk by ID.

        Args:
            point_id: Qdrant point ID

        Returns:
            Document chunk dict or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )

            if points:
                return self._point_to_dict(points[0])
            else:
                return None

        except Exception as e:
            logger.error(f"Error retrieving point {point_id}: {e}")
            return None

    async def batch_search(
        self,
        query_vectors: list[list[float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """
        Perform batch semantic search for multiple queries.

        Args:
            query_vectors: List of embedding vectors
            top_k: Number of results per query
            filters: Metadata filters applied to all queries

        Returns:
            List of result lists, one per query
        """
        try:
            qdrant_filter = self._build_filter(filters) if filters else None

            # Batch search
            if hasattr(self.client, "search_batch"):
                batch_results = self.client.search_batch(
                    collection_name=self.collection_name,
                    requests=[
                        {
                            "vector": vec,
                            "limit": top_k,
                            "filter": qdrant_filter,
                            "with_payload": True,
                            "with_vectors": False,
                        }
                        for vec in query_vectors
                    ],
                )
            else:
                requests = [
                    qmodels.QueryRequest(
                        query=vec,
                        limit=top_k,
                        filter=qdrant_filter,
                        with_payload=True,
                        with_vector=False,
                    )
                    for vec in query_vectors
                ]
                batch_results = self.client.query_batch_points(
                    collection_name=self.collection_name,
                    requests=requests,
                )

            # Convert results
            results = [
                [self._point_to_dict(point) for point in batch.points]
                for batch in batch_results
            ]

            logger.info(f"Batch search completed for {len(query_vectors)} queries")

            return results

        except Exception as e:
            logger.error(f"Error during batch search: {e}")
            raise

    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Collection info including count, vector size, etc.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Qdrant is accessible and healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    # ===== Helper Methods =====

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """
        Build Qdrant Filter from dict of metadata filters.

        Args:
            filters: Dict like {"regulation": "EU AI Act", "domain": "recruitment"}

        Returns:
            Qdrant Filter object
        """
        conditions = []

        field_map = {
            "regulation": "regulation_name",
            "celex": "celex_id",
            "article": "article_number",
            "domain": "domains",
        }

        for key, value in filters.items():
            if value is not None:
                mapped_key = field_map.get(key, key)
                # Handle different value types
                if isinstance(value, list):
                    # Match any value in list
                    for v in value:
                        conditions.append(
                            FieldCondition(
                                key=mapped_key,
                                match=MatchValue(value=v),
                            )
                        )
                else:
                    # Exact match
                    conditions.append(
                        FieldCondition(
                            key=mapped_key,
                            match=MatchValue(value=value),
                        )
                    )

        if conditions:
            return Filter(must=conditions)
        else:
            return None

    def _point_to_dict(self, point: ScoredPoint | PointStruct) -> dict[str, Any]:
        """
        Convert Qdrant point to dict format.

        Args:
            point: Qdrant ScoredPoint or PointStruct

        Returns:
            Dict with id, score, content, and metadata
        """
        payload = point.payload or {}

        content = payload.get("content") or payload.get("text") or ""
        metadata = payload.get("metadata") or payload

        result = {
            "id": str(point.id),
            "score": getattr(point, "score", None),
            "content": content,
            "metadata": metadata,
        }

        # Flatten metadata for easier access
        metadata = result["metadata"]
        result.update({
            "regulation": metadata.get("regulation") or metadata.get("regulation_name"),
            "article": metadata.get("article") or metadata.get("article_number"),
            "paragraph": metadata.get("paragraph") or metadata.get("paragraph_index"),
            "celex": metadata.get("celex") or metadata.get("celex_id"),
            "domain": metadata.get("domain") or metadata.get("domains"),
            "risk_category": metadata.get("risk_category"),
            "effective_date": metadata.get("effective_date"),
            "chunk_index": metadata.get("chunk_index"),
        })

        return result


# ===== Singleton instance =====

_qdrant_store: QdrantVectorStore | None = None


def get_qdrant_store() -> QdrantVectorStore:
    """
    Get singleton Qdrant store instance.

    Returns:
        QdrantVectorStore instance
    """
    global _qdrant_store

    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore()

    return _qdrant_store
