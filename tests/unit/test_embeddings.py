"""
Unit Tests for Embedding Generator

Tests OpenAI embedding generation with mocked API calls.
"""

import pytest
from unittest.mock import Mock, patch

from embeddings.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Tests for embedding generation."""

    @pytest.fixture
    def generator(self):
        """Create embedding generator with mocked client."""
        with patch('embeddings.embedding_generator.OpenAI') as mock_openai:
            # Mock the client initialization
            mock_client = Mock()
            mock_openai.return_value = mock_client

            # Mock the initial test embedding call
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1024)]
            mock_client.embeddings.create.return_value = mock_response

            generator = EmbeddingGenerator(
                model_name="text-embedding-3-large",
                batch_size=10,
                dimensions=1024,
                show_progress=False,
            )

            yield generator

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.model_name == "text-embedding-3-large"
        assert generator.batch_size == 10
        assert generator.embedding_dim == 1024

    def test_generate_embeddings_single_chunk(self, generator, sample_chunk):
        """Test generating embeddings for a single chunk."""
        chunks = [sample_chunk]

        # Mock API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]
        generator.client.embeddings.create.return_value = mock_response

        result = generator.generate_embeddings(chunks)

        assert len(result) == 1
        assert result[0].embedding is not None
        assert len(result[0].embedding) == 1024
        generator.client.embeddings.create.assert_called()

    def test_generate_embeddings_multiple_chunks(self, generator, sample_chunks):
        """Test generating embeddings for multiple chunks."""
        # Mock API response
        mock_embeddings = [Mock(embedding=[0.1] * 1024) for _ in sample_chunks]
        mock_response = Mock()
        mock_response.data = mock_embeddings
        generator.client.embeddings.create.return_value = mock_response

        result = generator.generate_embeddings(sample_chunks)

        assert len(result) == len(sample_chunks)
        for chunk in result:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024

    def test_generate_query_embedding(self, generator):
        """Test generating embedding for a query string."""
        query = "What are the GDPR requirements?"

        # Mock API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.2] * 1024)]
        generator.client.embeddings.create.return_value = mock_response

        embedding = generator.generate_query_embedding(query)

        assert embedding is not None
        assert len(embedding) == 1024
        generator.client.embeddings.create.assert_called()

    def test_embedding_preserves_chunk_metadata(self, generator, sample_chunk):
        """Test that embedding generation preserves chunk metadata."""
        # Mock API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1024)]
        generator.client.embeddings.create.return_value = mock_response

        original_metadata = sample_chunk.metadata.model_copy()
        result = generator.generate_embeddings([sample_chunk])

        # Verify metadata is preserved
        assert result[0].metadata.regulation_name == original_metadata.regulation_name
        assert result[0].metadata.celex_id == original_metadata.celex_id
        assert result[0].metadata.article_number == original_metadata.article_number

    def test_batch_processing(self, generator, generate_chunks):
        """Test embeddings are generated in batches."""
        # Create more chunks than batch_size
        chunks = generate_chunks(count=25, with_embeddings=False)

        # Mock API response to return appropriate number of embeddings per batch
        def create_mock_response(**kwargs):
            input_data = kwargs.get('input', [])
            num_texts = len(input_data) if isinstance(input_data, list) else 1
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1024) for _ in range(num_texts)]
            return mock_response

        generator.client.embeddings.create.side_effect = create_mock_response

        # Reset call count after fixture setup
        generator.client.embeddings.create.reset_mock()

        result = generator.generate_embeddings(chunks)

        assert len(result) == 25
        # Should have made 3 calls (10 + 10 + 5)
        assert generator.client.embeddings.create.call_count == 3

    def test_empty_chunks_list(self, generator):
        """Test handling empty chunks list."""
        result = generator.generate_embeddings([])
        assert result == []

    @pytest.mark.unit
    def test_api_error_handling(self, generator, sample_chunk):
        """Test handling API errors gracefully."""
        from openai import OpenAIError

        # Mock API to raise error
        generator.client.embeddings.create.side_effect = OpenAIError("API Error")

        # Should raise the error (not handle it silently)
        with pytest.raises(OpenAIError):
            generator.generate_embeddings([sample_chunk])
