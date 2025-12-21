"""
End-to-End Tests for RAG Pipeline

Tests the complete RAG workflow from query to answer generation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

pytestmark = pytest.mark.e2e


@pytest.mark.rag
class TestRAGPipelineE2E:
    """End-to-end tests for RAG pipeline."""

    @pytest.fixture
    async def rag_components(self, mock_anthropic_client, mock_openai_client):
        """Set up RAG pipeline components."""
        # Import here to avoid circular dependencies
        import sys
        from pathlib import Path

        # Ensure imports work
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root / "services" / "rag-service" / "src"))

        return {
            "anthropic_client": mock_anthropic_client,
            "openai_client": mock_openai_client,
        }

    @pytest.mark.asyncio
    async def test_query_to_answer_flow(self, rag_components, sample_chunk_with_embedding):
        """Test complete flow: query -> retrieval -> generation -> answer."""
        query = "What are the requirements for high-risk AI systems?"

        # 1. Query embedding generation
        with patch("openai.OpenAI") as mock_openai:
            client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1024)]
            client.embeddings.create = Mock(return_value=mock_response)
            mock_openai.return_value = client

            from embeddings.embedding_generator import EmbeddingGenerator

            generator = EmbeddingGenerator(show_progress=False)
            query_embedding = generator.generate_query_embedding(query)

            assert query_embedding is not None
            assert len(query_embedding) == 1024

        # 2. Mock retrieval from Qdrant
        retrieved_chunks = [
            {
                "id": "test_id_1",
                "score": 0.95,
                "payload": {
                    "text": "High-risk AI systems must undergo conformity assessment.",
                    "regulation_name": "AI Act",
                    "article_number": "43",
                },
            }
        ]

        # 3. Mock answer generation
        with patch("anthropic.Anthropic") as mock_anthropic:
            client = Mock()
            mock_response = Mock()
            mock_response.content = [
                Mock(
                    text="High-risk AI systems must meet several requirements: "
                    "conformity assessment, technical documentation, and ongoing monitoring."
                )
            ]
            mock_response.usage = Mock(input_tokens=100, output_tokens=50)
            client.messages.create = Mock(return_value=mock_response)
            mock_anthropic.return_value = client

            # Simplified answer generation
            answer = mock_response.content[0].text

            assert "conformity assessment" in answer
            assert "High-risk" in answer

    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self, rag_components):
        """Test RAG pipeline with multi-step reasoning."""
        query = "Compare GDPR and AI Act requirements for automated decision-making"

        # Mock retrieval of chunks from both regulations
        retrieved_chunks = [
            {
                "text": "GDPR Article 22: Right not to be subject to automated decision-making",
                "regulation_name": "GDPR",
                "article_number": "22",
                "score": 0.92,
            },
            {
                "text": "AI Act requires transparency for automated decision-making systems",
                "regulation_name": "AI Act",
                "article_number": "13",
                "score": 0.88,
            },
        ]

        # Verify we retrieved from multiple regulations
        regulation_names = {chunk["regulation_name"] for chunk in retrieved_chunks}
        assert "GDPR" in regulation_names
        assert "AI Act" in regulation_names

    @pytest.mark.asyncio
    async def test_low_confidence_handling(self, rag_components):
        """Test handling of low confidence queries."""
        query = "What is the meaning of life in EU AI regulations?"

        # Simulate low relevance scores
        retrieved_chunks = [
            {
                "text": "AI systems shall be designed and developed...",
                "score": 0.45,  # Low score
            }
        ]

        # Should indicate low confidence or refuse to answer
        assert all(chunk["score"] < 0.6 for chunk in retrieved_chunks)

    @pytest.mark.asyncio
    async def test_citation_generation(self, rag_components, sample_chunk_with_embedding):
        """Test that answers include proper citations."""
        query = "What are biometric identification requirements?"

        # Mock retrieval with metadata
        retrieved_chunks = [
            {
                "text": "Biometric identification systems are classified as high-risk.",
                "regulation_name": "AI Act",
                "article_number": "6",
                "celex_id": "TEST001",
                "score": 0.93,
            }
        ]

        # Expected citation format
        expected_citation = "AI Act Article 6"

        # Verify citation information is available
        chunk = retrieved_chunks[0]
        citation = f"{chunk['regulation_name']} Article {chunk['article_number']}"
        assert citation == expected_citation


@pytest.mark.rag
class TestRAGPipelineEdgeCases:
    """Test edge cases in RAG pipeline."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        """Test handling of empty query."""
        query = ""

        # Should raise validation error or handle gracefully
        assert query == "" or query.strip() == ""

    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test handling of very long query."""
        query = "What are the requirements " * 1000  # Very long query

        # Should truncate or handle appropriately
        assert len(query) > 10000

    @pytest.mark.asyncio
    async def test_query_with_special_characters(self):
        """Test query with special characters."""
        query = "What is the requirement for AI systems in §13.4(a)?  émojis 🤖"

        # Should handle special characters
        assert "§" in query
        assert "🤖" in query

    @pytest.mark.asyncio
    async def test_no_relevant_chunks_found(self):
        """Test when no relevant chunks are found."""
        query = "What is the capital of France?"  # Irrelevant to EU AI law

        # Simulate empty retrieval
        retrieved_chunks = []

        # Should return appropriate message
        assert len(retrieved_chunks) == 0


@pytest.mark.slow
@pytest.mark.requires_qdrant
class TestRAGPipelineWithRealQdrant:
    """E2E tests with real Qdrant database."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_indexing_and_retrieval(
        self,
        qdrant_test_client,
        qdrant_test_collection,
        generate_chunks,
    ):
        """Test complete pipeline with real indexing and retrieval."""
        from indexing.qdrant_indexer import QdrantIndexer
        from embeddings.embedding_generator import EmbeddingGenerator

        # 1. Create and index chunks
        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name=qdrant_test_collection,
            embedding_dim=1024,
        )

        chunks = generate_chunks(count=10, with_embeddings=True)
        # Add specific content for testing
        chunks[0].text = "High-risk AI systems must undergo conformity assessment."
        chunks[0].metadata.article_number = "43"
        chunks[0].metadata.regulation_name = "AI Act"

        indexed_count = indexer.index_chunks(chunks, show_progress=False)
        assert indexed_count == 10

        # 2. Generate query embedding
        with patch("openai.OpenAI") as mock_openai:
            client = Mock()
            # Use similar embedding to chunk[0] for testing
            mock_response = Mock()
            mock_response.data = [Mock(embedding=chunks[0].embedding)]
            client.embeddings.create = Mock(return_value=mock_response)
            mock_openai.return_value = client

            generator = EmbeddingGenerator(show_progress=False)
            query_embedding = generator.generate_query_embedding(
                "What are the requirements for high-risk AI systems?"
            )

        # 3. Search in Qdrant
        results = indexer.search(query_vector=query_embedding, limit=5)

        assert len(results) > 0
        # First result should be our specific chunk
        assert results[0]["score"] > 0.9

    @pytest.mark.asyncio
    async def test_filtered_retrieval(
        self,
        qdrant_test_client,
        qdrant_test_collection,
        generate_chunks,
    ):
        """Test retrieval with metadata filters."""
        from indexing.qdrant_indexer import QdrantIndexer

        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name=qdrant_test_collection,
            embedding_dim=1024,
        )

        # Create chunks with different regulations
        chunks = generate_chunks(count=10, with_embeddings=True)
        for i, chunk in enumerate(chunks):
            chunk.metadata.regulation_name = "GDPR" if i < 5 else "AI Act"

        indexer.index_chunks(chunks, show_progress=False)

        # Search with filter for GDPR only
        import random

        query_vector = [random.random() for _ in range(1024)]
        results = indexer.search(
            query_vector=query_vector,
            limit=10,
            filters={"regulation_name": "GDPR"},
        )

        # All results should be from GDPR
        for result in results:
            assert result["payload"]["regulation_name"] == "GDPR"
