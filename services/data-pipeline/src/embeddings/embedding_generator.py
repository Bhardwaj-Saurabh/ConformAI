"""
Embedding Generator for Legal Documents

Generates dense vector embeddings using OpenAI's embedding models.

Supported OpenAI models:
- text-embedding-3-large (3072 dim) - Best performance, adjustable dimensions
- text-embedding-3-small (1536 dim) - Good balance of quality and cost
- text-embedding-ada-002 (1536 dim) - Legacy model
"""

import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from shared.config import get_settings
from shared.models import Chunk
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingGenerator:
    """Generator for semantic embeddings of legal text."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
        show_progress: bool = True,
        dimensions: int | None = None,
    ):
        """
        Initialize OpenAI embedding generator.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            batch_size: Batch size for API requests
            show_progress: Show progress bar during encoding
            dimensions: Output dimension (only for text-embedding-3-* models)
        """
        self.model_name = model_name or "text-embedding-3-large"
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.dimensions = dimensions

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)

        # Set embedding dimension based on model
        model_dims = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }

        if self.dimensions:
            # Custom dimension (only for v3 models)
            if "text-embedding-3" not in self.model_name:
                logger.warning(
                    f"Custom dimensions not supported for {self.model_name}. Using default."
                )
                self.embedding_dim = model_dims.get(self.model_name, 1536)
            else:
                self.embedding_dim = self.dimensions
        else:
            self.embedding_dim = model_dims.get(self.model_name, 1536)

        logger.info(f"Initializing OpenAI embedding model: {self.model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Test API connection
        try:
            test_embedding = self.client.embeddings.create(
                model=self.model_name, input=["test"], dimensions=self.dimensions
            )
            logger.info("✓ OpenAI API connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {str(e)}")
            raise

    def generate_embeddings(
        self, chunks: list[Chunk], normalize: bool = True
    ) -> list[Chunk]:
        """
        Generate embeddings for a list of chunks using OpenAI API.

        Args:
            chunks: List of chunks without embeddings
            normalize: Whether to normalize embeddings (Note: OpenAI embeddings are pre-normalized)

        Returns:
            Same chunks with embeddings added
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []

        logger.info(f"Generating OpenAI embeddings for {len(chunks)} chunks...")

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        start_time = time.time()
        all_embeddings = []

        # Process in batches to respect API rate limits
        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]

        iterator = enumerate(batches)
        if self.show_progress:
            iterator = tqdm(iterator, total=len(batches), desc="Generating embeddings")

        for batch_idx, batch_texts in iterator:
            try:
                batch_start = time.time()

                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch_texts, dimensions=self.dimensions
                )

                batch_duration = (time.time() - batch_start) * 1000

                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Log batch performance
                logger.debug(
                    f"Batch {batch_idx + 1}/{len(batches)} completed",
                    extra={
                        "batch_index": batch_idx,
                        "batch_size": len(batch_texts),
                        "duration_ms": batch_duration,
                        "embeddings_per_second": len(batch_texts) / (batch_duration / 1000) if batch_duration > 0 else 0,
                        "model": self.model_name,
                    },
                )

                # Small delay to avoid rate limits
                if batch_idx < len(batches) - 1:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(
                    f"Failed to generate embeddings for batch {batch_idx}: {str(e)}",
                    extra={
                        "batch_index": batch_idx,
                        "batch_size": len(batch_texts),
                        "model": self.model_name,
                        "error": str(e),
                    },
                )
                raise

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {len(chunks)} embeddings in {elapsed:.2f}s "
            f"({len(chunks)/elapsed:.1f} chunks/sec)"
        )

        return chunks

    def generate_single_embedding(self, text: str, normalize: bool = True) -> list[float]:
        """
        Generate embedding for a single text using OpenAI API.

        Args:
            text: Text to embed
            normalize: Not used (OpenAI embeddings are pre-normalized)

        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name, input=[text], dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate single embedding: {str(e)}")
            raise

    def generate_query_embedding(self, query: str, normalize: bool = True) -> list[float]:
        """
        Generate embedding for a search query.

        Note: OpenAI's embedding models don't require special query prefixes.

        Args:
            query: Search query
            normalize: Not used (OpenAI embeddings are pre-normalized)

        Returns:
            Query embedding vector
        """
        return self.generate_single_embedding(query, normalize=normalize)

    def benchmark(self, num_samples: int = 100, text_length: int = 512):
        """
        Benchmark OpenAI embedding API performance.

        Args:
            num_samples: Number of sample texts to generate
            text_length: Approximate length of each text in characters

        Returns:
            Dictionary with benchmark results
        """
        import random
        import string

        logger.info(f"Running benchmark with {num_samples} samples...")

        # Generate random texts
        def random_text(length: int) -> str:
            words = [
                "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
                for _ in range(length // 6)
            ]
            return " ".join(words)

        texts = [random_text(text_length) for _ in range(num_samples)]

        # Benchmark
        start_time = time.time()

        # Process in batches
        batches = [
            texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)
        ]

        for batch in batches:
            self.client.embeddings.create(
                model=self.model_name, input=batch, dimensions=self.dimensions
            )
            time.sleep(0.1)  # Respect rate limits

        elapsed = time.time() - start_time

        results = {
            "num_samples": num_samples,
            "text_length": text_length,
            "total_time": elapsed,
            "samples_per_second": num_samples / elapsed,
            "time_per_sample": elapsed / num_samples,
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "api": "OpenAI",
        }

        logger.info("Benchmark results:")
        logger.info(f"  Total time: {elapsed:.2f}s")
        logger.info(f"  Throughput: {results['samples_per_second']:.1f} samples/sec")
        logger.info(f"  Latency: {results['time_per_sample']*1000:.1f}ms per sample")

        return results

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "api": "OpenAI",
            "dimensions": self.dimensions,
        }


# Example usage
if __name__ == "__main__":
    from shared.models import Chunk, ChunkMetadata

    # Initialize OpenAI embedding generator
    generator = EmbeddingGenerator(
        model_name="text-embedding-3-large",
        batch_size=100,
        dimensions=1024,  # Reduce from 3072 to 1024 for efficiency
    )

    # Print model info
    info = generator.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Create sample chunks
    sample_chunks = [
        Chunk(
            text="The data subject shall have the right not to be subject to automated decision-making.",
            metadata=ChunkMetadata(
                regulation_name="GDPR",
                celex_id="32016R0679",
                article_number="22",
                article_title="Automated decision-making",
            ),
        ),
        Chunk(
            text="High-risk AI systems shall be subject to conformity assessment before being placed on the market.",
            metadata=ChunkMetadata(
                regulation_name="AI Act",
                celex_id="unknown",
                article_number="43",
                article_title="Conformity assessment",
            ),
        ),
    ]

    # Generate embeddings
    chunks_with_embeddings = generator.generate_embeddings(sample_chunks)

    print(f"\nGenerated {len(chunks_with_embeddings)} embeddings")
    print(f"Embedding dimension: {len(chunks_with_embeddings[0].embedding)}")
    print(f"First embedding (first 5 values): {chunks_with_embeddings[0].embedding[:5]}")

    # Generate query embedding
    query = "What are the requirements for high-risk AI systems?"
    query_embedding = generator.generate_query_embedding(query)
    print(f"\nQuery embedding dimension: {len(query_embedding)}")
    print(f"Query embedding (first 5 values): {query_embedding[:5]}")

    # Run benchmark (optional)
    # results = generator.benchmark(num_samples=50)
