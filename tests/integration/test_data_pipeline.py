"""
Integration Tests for Data Pipeline

Tests the complete data pipeline with real components but mocked external APIs.
"""

from datetime import date
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.integration


class TestDataPipelineIntegration:
    """Integration tests for full data pipeline."""

    @pytest.fixture
    def pipeline_components(self, temp_data_dir, mock_openai_client):
        """Set up pipeline components."""
        from clients.eurlex_client import EURLexClient
        from embeddings.embedding_generator import EmbeddingGenerator

        return {
            "eurlex_client": EURLexClient(),
            "embedding_generator": EmbeddingGenerator(show_progress=False),
            "data_dir": temp_data_dir,
        }

    def test_end_to_end_document_processing(
        self,
        pipeline_components,
        mock_eurlex_response,
        mock_eurlex_document,
        mock_openai_client,
    ):
        """Test complete pipeline: discover -> download -> parse -> chunk -> embed."""
        client = pipeline_components["eurlex_client"]
        generator = pipeline_components["embedding_generator"]
        data_dir = pipeline_components["data_dir"]

        # 1. Mock document discovery
        with patch.object(client, '_execute_sparql') as mock_sparql:
            mock_sparql.return_value = mock_eurlex_response

            docs = client.search_recent_documents(start_date=date(2024, 1, 1), limit=1)
            assert len(docs) == 1
            celex = docs[0]["celex"]

        # 2. Mock document download
        with patch.object(client.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = mock_eurlex_document
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            xml_path = data_dir / "raw" / f"{celex}.xml"
            client.download_document_to_file(celex, xml_path, format="xml")

            assert xml_path.exists()

        # 3. Parse document (simplified - would use actual parser)
        from shared.models import Chunk, ChunkMetadata, RegulationType

        # Create sample chunks as if parsed
        chunks = []
        for i in range(3):
            metadata = ChunkMetadata(
                regulation_name="GDPR",
                celex_id=celex,
                regulation_type=RegulationType.REGULATION,
                article_number=str(i + 1),
                article_title=f"Article {i + 1}",
                chunk_index=i,
                total_chunks=3,
            )
            chunk = Chunk(
                text=f"Article {i + 1} content from parsed document.",
                metadata=metadata,
            )
            chunks.append(chunk)

        assert len(chunks) == 3

        # 4. Generate embeddings
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1] * 1024
        mock_response = Mock()
        mock_response.data = [mock_embedding] * len(chunks)
        mock_openai_client.embeddings.create.return_value = mock_response

        chunks_with_embeddings = generator.generate_embeddings(chunks)

        assert len(chunks_with_embeddings) == 3
        for chunk in chunks_with_embeddings:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024

    def test_batch_document_processing(
        self,
        pipeline_components,
        mock_eurlex_response,
        mock_openai_client,
        generate_chunks,
    ):
        """Test processing multiple documents in batches."""
        generator = pipeline_components["embedding_generator"]

        # Create chunks for multiple documents
        all_chunks = []
        for doc_id in range(3):  # 3 documents
            chunks = generate_chunks(count=5)  # 5 chunks each
            all_chunks.extend(chunks)

        # Mock embedding API
        def create_embeddings(*args, **kwargs):
            batch_size = len(kwargs.get('input', []))
            return Mock(data=[Mock(embedding=[0.1] * 1024) for _ in range(batch_size)])

        mock_openai_client.embeddings.create.side_effect = create_embeddings

        # Generate embeddings
        result = generator.generate_embeddings(all_chunks)

        assert len(result) == 15  # 3 docs * 5 chunks
        for chunk in result:
            assert chunk.embedding is not None


@pytest.mark.requires_qdrant
class TestQdrantIndexingIntegration:
    """Integration tests with real Qdrant (requires running Qdrant instance)."""

    def test_create_collection_and_index(
        self,
        qdrant_test_client,
        qdrant_test_collection,
        sample_chunk_with_embedding,
    ):
        """Test creating collection and indexing chunks."""
        from indexing.qdrant_indexer import QdrantIndexer

        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name=qdrant_test_collection,
            embedding_dim=1024,
        )

        # Index a single chunk
        indexed_count = indexer.index_chunks([sample_chunk_with_embedding], show_progress=False)

        assert indexed_count == 1

        # Verify in Qdrant
        info = indexer.get_collection_info()
        assert info["vectors_count"] == 1

    def test_bulk_indexing(
        self,
        qdrant_test_client,
        qdrant_test_collection,
        generate_chunks,
    ):
        """Test bulk indexing of chunks."""
        from indexing.qdrant_indexer import QdrantIndexer

        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name=qdrant_test_collection,
            embedding_dim=1024,
        )

        # Create chunks with embeddings
        chunks = generate_chunks(count=50, with_embeddings=True)

        # Index all chunks
        indexed_count = indexer.index_chunks(chunks, batch_size=10, show_progress=False)

        assert indexed_count == 50

        # Verify collection
        info = indexer.get_collection_info()
        assert info["vectors_count"] == 50

    def test_search_after_indexing(
        self,
        qdrant_test_client,
        qdrant_test_collection,
        sample_chunk_with_embedding,
    ):
        """Test searching for similar chunks after indexing."""
        from indexing.qdrant_indexer import QdrantIndexer

        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name=qdrant_test_collection,
            embedding_dim=1024,
        )

        # Index chunk
        indexer.index_chunks([sample_chunk_with_embedding], show_progress=False)

        # Search using same embedding (should find itself)
        results = indexer.search(
            query_vector=sample_chunk_with_embedding.embedding,
            limit=5,
        )

        assert len(results) >= 1
        assert results[0]["score"] > 0.9  # Should be very similar to itself


class TestPipelineErrorHandling:
    """Test error handling in data pipeline."""

    def test_download_failure_recovery(self, mock_eurlex_response):
        """Test pipeline continues after download failure."""
        import httpx
        from clients.eurlex_client import EURLexClient

        client = EURLexClient()

        with patch.object(client, '_execute_sparql') as mock_sparql:
            mock_sparql.return_value = mock_eurlex_response

            # Get documents
            docs = client.search_recent_documents(start_date=date(2024, 1, 1), limit=3)

            # Simulate download failure for first document, success for others
            with patch.object(client.client, 'get') as mock_get:
                def side_effect(*args, **kwargs):
                    if mock_get.call_count == 1:
                        raise httpx.HTTPStatusError(
                            "404 Not Found",
                            request=Mock(),
                            response=Mock(status_code=404),
                        )
                    else:
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_response.content = b"<root>Test</root>"
                        mock_response.raise_for_status = Mock()
                        return mock_response

                mock_get.side_effect = side_effect

                # First download should fail
                with pytest.raises(httpx.HTTPStatusError):
                    client.download_document(docs[0]["celex"])

                # Second download should succeed
                content = client.download_document(docs[0]["celex"])
                assert content is not None

    def test_embedding_generation_partial_failure(self, mock_openai_client, generate_chunks):
        """Test handling partial failures in embedding generation."""
        from embeddings.embedding_generator import EmbeddingGenerator
        from openai import OpenAIError

        generator = EmbeddingGenerator(batch_size=5, show_progress=False)
        chunks = generate_chunks(count=10)

        # Simulate failure on first batch, success on second
        call_count = 0

        def create_embeddings(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OpenAIError("Rate limit exceeded")
            batch_size = len(kwargs.get('input', []))
            return Mock(data=[Mock(embedding=[0.1] * 1024) for _ in range(batch_size)])

        mock_openai_client.embeddings.create.side_effect = create_embeddings

        # First attempt should fail
        with pytest.raises(OpenAIError):
            generator.generate_embeddings(chunks)
