"""
Qdrant Vector Database Indexer

Handles indexing of legal document chunks into Qdrant with rich metadata.

Features:
- Collection creation and management
- Batch indexing with progress tracking
- Metadata filtering support
- Version management for document updates
- Deduplication
"""

import hashlib
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

from shared.config import get_settings
from shared.models import Chunk
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class QdrantIndexer:
    """Indexer for Qdrant vector database."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
    ):
        """
        Initialize Qdrant indexer.

        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (optional)
            collection_name: Name of collection to use
            embedding_dim: Dimension of embeddings
        """
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim or settings.embedding_dimension

        logger.info(f"Connecting to Qdrant at {self.url}")

        # Initialize client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key if self.api_key else None,
        )

        # Verify connection
        try:
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def create_collection(
        self,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
        distance: Distance = Distance.COSINE,
        recreate: bool = False,
    ):
        """
        Create Qdrant collection with metadata schema.

        Args:
            collection_name: Name of collection (uses default if None)
            embedding_dim: Dimension of embeddings
            distance: Distance metric (COSINE, DOT, EUCLID)
            recreate: If True, delete existing collection and recreate

        Raises:
            ValueError: If collection already exists and recreate=False
        """
        collection_name = collection_name or self.collection_name
        embedding_dim = embedding_dim or self.embedding_dim

        logger.info(f"Creating collection: {collection_name} (dim={embedding_dim})")

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            if recreate:
                logger.warning(f"Collection {collection_name} exists. Deleting...")
                self.client.delete_collection(collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists")
                return

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=distance,
            ),
        )

        # Create payload indexes for fast filtering
        self._create_payload_indexes(collection_name)

        logger.info(f"Collection {collection_name} created successfully")

    def _create_payload_indexes(self, collection_name: str):
        """Create indexes on payload fields for fast filtering."""
        # Index fields that will be used for filtering
        indexed_fields = [
            "regulation_name",
            "celex_id",
            "article_number",
            "domains",
            "risk_category",
            "version",
            "effective_date",
        ]

        for field in indexed_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword",  # Exact match
                )
                logger.debug(f"Created index on field: {field}")
            except Exception as e:
                logger.warning(f"Failed to create index on {field}: {e}")

    def index_chunks(
        self,
        chunks: list[Chunk],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Index chunks into Qdrant.

        Args:
            chunks: List of chunks with embeddings
            batch_size: Batch size for indexing
            show_progress: Show progress bar

        Returns:
            Number of chunks indexed

        Raises:
            ValueError: If chunks don't have embeddings
        """
        if not chunks:
            logger.warning("No chunks to index")
            return 0

        # Validate embeddings
        missing_embeddings = sum(1 for chunk in chunks if chunk.embedding is None)
        if missing_embeddings > 0:
            raise ValueError(f"{missing_embeddings} chunks are missing embeddings")

        logger.info(f"Indexing {len(chunks)} chunks to collection '{self.collection_name}'...")

        # Convert chunks to Qdrant points
        points = []
        for chunk in chunks:
            point = self._chunk_to_point(chunk)
            points.append(point)

        # Index in batches
        indexed_count = 0
        start_time = time.time()

        iterator = range(0, len(points), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Indexing batches")

        for i in iterator:
            batch = points[i : i + batch_size]

            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                indexed_count += len(batch)
            except Exception as e:
                logger.error(f"Failed to index batch {i//batch_size}: {str(e)}")
                raise

        elapsed = time.time() - start_time
        logger.info(
            f"Indexed {indexed_count} chunks in {elapsed:.2f}s "
            f"({indexed_count/elapsed:.1f} chunks/sec)"
        )

        return indexed_count

    def _chunk_to_point(self, chunk: Chunk) -> PointStruct:
        """
        Convert chunk to Qdrant point.

        Args:
            chunk: Chunk with embedding and metadata

        Returns:
            Qdrant point structure
        """
        # Generate deterministic ID based on content and metadata
        id_string = f"{chunk.metadata.celex_id}:{chunk.metadata.article_number}:{chunk.metadata.chunk_index}"
        point_id = hashlib.md5(id_string.encode()).hexdigest()

        # Convert metadata to payload
        payload = {
            # Document identifiers
            "regulation_name": chunk.metadata.regulation_name,
            "celex_id": chunk.metadata.celex_id,
            "regulation_type": chunk.metadata.regulation_type.value,
            # Structure
            "chapter_number": chunk.metadata.chapter_number,
            "chapter_title": chunk.metadata.chapter_title,
            "article_number": chunk.metadata.article_number,
            "article_title": chunk.metadata.article_title,
            "paragraph_index": chunk.metadata.paragraph_index,
            # Chunk info
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            # Temporal
            "effective_date": (
                chunk.metadata.effective_date.isoformat()
                if chunk.metadata.effective_date
                else None
            ),
            "version": chunk.metadata.version,
            # Classification
            "domains": [domain.value for domain in chunk.metadata.domains],
            "risk_category": (
                chunk.metadata.risk_category.value if chunk.metadata.risk_category else None
            ),
            # Source
            "source_url": chunk.metadata.source_url,
            "last_updated": chunk.metadata.last_updated.isoformat(),
            # Content
            "text": chunk.text,
        }

        return PointStruct(
            id=point_id,
            vector=chunk.embedding,
            payload=payload,
        )

    def delete_by_celex(self, celex_id: str) -> int:
        """
        Delete all chunks for a specific regulation.

        Useful when updating a regulation to a new version.

        Args:
            celex_id: CELEX identifier

        Returns:
            Number of points deleted
        """
        logger.info(f"Deleting chunks for CELEX {celex_id}")

        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="celex_id",
                    match=MatchValue(value=celex_id),
                )
            ]
        )

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_condition,
        )

        logger.info(f"Deleted points for {celex_id}")
        return result

    def count_documents(self, filters: dict[str, Any] | None = None) -> int:
        """
        Count documents in collection.

        Args:
            filters: Optional filters to apply

        Returns:
            Document count
        """
        if filters:
            filter_condition = self._build_filter(filters)
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=filter_condition,
            )
            return result.count
        else:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count

    def get_collection_info(self) -> dict[str, Any]:
        """Get collection information."""
        collection = self.client.get_collection(self.collection_name)

        return {
            "name": self.collection_name,
            "vectors_count": collection.points_count,
            "segments_count": collection.segments_count,
            "status": collection.status,
            "vector_size": collection.config.params.vectors.size,
            "distance": collection.config.params.vectors.distance,
        }

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []

        for key, value in filters.items():
            if value is not None:
                if isinstance(value, list):
                    # Multiple values - OR condition
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(any=value),
                        )
                    )
                else:
                    # Single value
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )

        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score

        Returns:
            List of search results with scores and payloads
        """
        filter_condition = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_condition,
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    def close(self):
        """Close Qdrant client connection."""
        logger.info("Closing Qdrant connection")
        self.client.close()


# Example usage
if __name__ == "__main__":
    from shared.models import Chunk, ChunkMetadata, RegulationType

    # Initialize indexer
    indexer = QdrantIndexer(
        url="http://localhost:6333",
        collection_name="eu_legal_documents_dev",
        embedding_dim=1024,
    )

    # Create collection
    indexer.create_collection(recreate=True)

    # Create sample chunks with fake embeddings
    import random

    sample_chunks = [
        Chunk(
            text="The data subject shall have the right to automated decision-making.",
            metadata=ChunkMetadata(
                regulation_name="GDPR",
                celex_id="32016R0679",
                regulation_type=RegulationType.REGULATION,
                article_number="22",
                article_title="Automated decision-making",
                chunk_index=0,
                total_chunks=1,
            ),
            embedding=[random.random() for _ in range(1024)],
        ),
    ]

    # Index chunks
    indexed = indexer.index_chunks(sample_chunks)
    print(f"Indexed {indexed} chunks")

    # Get collection info
    info = indexer.get_collection_info()
    print("\nCollection Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Search
    query_vec = [random.random() for _ in range(1024)]
    results = indexer.search(query_vec, limit=5)
    print(f"\nFound {len(results)} results")

    indexer.close()
