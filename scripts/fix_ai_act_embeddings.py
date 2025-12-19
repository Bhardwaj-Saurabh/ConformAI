#!/usr/bin/env python3
"""
Fix EU AI Act embeddings and index them into Qdrant.

This script:
1. Loads the AI Act chunks that were successfully created
2. Generates embeddings for them
3. Indexes them into Qdrant
"""

import pickle
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
data_pipeline_root = project_root / "services" / "data-pipeline" / "src"
retrieval_service_root = project_root / "services" / "retrieval-service" / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(data_pipeline_root))
sys.path.insert(0, str(retrieval_service_root))

from embeddings import EmbeddingGenerator
from indexing import QdrantIndexer
from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


def main():
    """Re-generate AI Act embeddings and index them."""

    # Paths
    chunks_file = project_root / "data" / "processed" / "chunks" / "52021PC0206_chunks.pkl"
    embeddings_file = project_root / "data" / "embeddings" / "52021PC0206_embedded.pkl"

    # 1. Load chunks
    logger.info(f"Loading AI Act chunks from {chunks_file}")
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"Loaded {len(chunks)} AI Act chunks")

    if not chunks:
        logger.error("No chunks found! Cannot proceed.")
        return 1

    # Filter out problematic chunks (empty or too short)
    original_count = len(chunks)
    chunks = [
        chunk for chunk in chunks
        if hasattr(chunk, 'text') and chunk.text and len(chunk.text.strip()) >= 10
    ]

    filtered_count = original_count - len(chunks)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} problematic chunks (empty or <10 chars)")
        logger.info(f"Proceeding with {len(chunks)} valid chunks")

    # 2. Generate embeddings
    logger.info("Generating embeddings using OpenAI text-embedding-3-large (dim=3072)")

    # Use smaller batch size to reduce chance of hitting API limits
    embedder = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=50,  # Reduced from 100 to be safer
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    # Validate chunks before embedding
    logger.info("Validating chunk texts...")
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, 'text') or not chunk.text:
            logger.error(f"Chunk {i} has no text! This shouldn't happen after filtering.")
            raise ValueError(f"Invalid chunk at index {i}")

        # Ensure text is a string
        if not isinstance(chunk.text, str):
            chunk.text = str(chunk.text)

    logger.info("✓ All chunks validated")

    try:
        chunks_with_embeddings = embedder.generate_embeddings(chunks)
        logger.info(f"✓ Successfully generated {len(chunks_with_embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        logger.error("Try running with smaller batch size or check OpenAI API status")
        raise

    # 3. Save embeddings
    logger.info(f"Saving embeddings to {embeddings_file}")
    with open(embeddings_file, "wb") as f:
        pickle.dump(chunks_with_embeddings, f)

    logger.info(f"✓ Saved {len(chunks_with_embeddings)} embedded chunks")

    # 4. Index into Qdrant
    logger.info("Indexing into Qdrant...")
    indexer = QdrantIndexer(
        collection_name=settings.qdrant_collection_name,
    )

    try:
        # Note: This will append to existing collection, not delete GDPR/Data Gov data
        indexer.index_chunks(chunks_with_embeddings)
        logger.info(f"✓ Successfully indexed {len(chunks_with_embeddings)} AI Act chunks into Qdrant")
    except Exception as e:
        logger.error(f"Failed to index chunks: {e}")
        raise

    # 5. Verify
    logger.info("\n=== Verification ===")
    from retrieval.qdrant_client import QdrantVectorStore

    store = QdrantVectorStore()
    collection_info = store.client.get_collection(settings.qdrant_collection_name)

    logger.info(f"Total points in collection: {collection_info.points_count}")
    logger.info(f"Expected: ~{124 + len(chunks_with_embeddings)} (124 existing + {len(chunks_with_embeddings)} new)")

    # Count AI Act documents
    points = store.client.scroll(
        collection_name=settings.qdrant_collection_name,
        scroll_filter={
            'must': [
                {
                    'key': 'regulation_name',
                    'match': {'value': 'EUR-Lex - 52021PC0206'}
                }
            ]
        },
        limit=1000,
        with_payload=False,
        with_vectors=False
    )

    logger.info(f"AI Act chunks in database: {len(points[0])}")

    if len(points[0]) > 0:
        logger.info("✅ SUCCESS! EU AI Act is now indexed and ready for retrieval.")
    else:
        logger.warning("⚠️ AI Act chunks not found in database. Check indexing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
