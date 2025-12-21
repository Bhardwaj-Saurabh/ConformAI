"""
Unit Tests for EUR-Lex Client

Tests EUR-Lex API client with mocked HTTP responses.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch, MagicMock

from clients.eurlex_client import EURLexClient
from shared.models import RegulationType


class TestEURLexClient:
    """Tests for EUR-Lex API client."""

    @pytest.fixture
    def client(self):
        """Create EUR-Lex client."""
        return EURLexClient()

    def test_client_initialization(self, client):
        """Test client initializes with default values."""
        assert client.sparql_endpoint == "https://publications.europa.eu/webapi/rdf/sparql"
        assert client.rest_base_url == "https://eur-lex.europa.eu"
        assert client.timeout == 60

    def test_execute_sparql_success(self, client, mock_eurlex_response):
        """Test successful SPARQL query execution."""
        with patch.object(client.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_eurlex_response
            mock_response.content = b'{"results": []}'  # Add content for len()
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            result = client._execute_sparql("SELECT * WHERE { ?s ?p ?o }")

            assert result == mock_eurlex_response
            mock_post.assert_called_once()

    def test_execute_sparql_failure(self, client):
        """Test SPARQL query handles HTTP errors."""
        import httpx

        with patch.object(client.client, 'post') as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "500 Server Error",
                request=Mock(),
                response=Mock(status_code=500),
            )

            with pytest.raises(httpx.HTTPStatusError):
                client._execute_sparql("SELECT * WHERE { ?s ?p ?o }")

    def test_search_recent_documents(self, client, mock_eurlex_response):
        """Test searching for recent documents."""
        with patch.object(client, '_execute_sparql') as mock_sparql:
            mock_sparql.return_value = mock_eurlex_response

            docs = client.search_recent_documents(
                start_date=date(2024, 1, 1),
                limit=10,
            )

            assert len(docs) == 1
            assert docs[0]["celex"] == "32016R0679"
            assert docs[0]["title"] == "General Data Protection Regulation"
            mock_sparql.assert_called_once()

    def test_search_ai_related_documents(self, client, mock_eurlex_response):
        """Test searching for AI-related documents."""
        with patch.object(client, 'search_recent_documents') as mock_search:
            mock_search.return_value = [{"celex": "TEST001", "title": "AI Act"}]

            docs = client.search_ai_related_documents(
                start_date=date(2021, 1, 1),
                limit=5,
            )

            assert len(docs) == 1
            mock_search.assert_called_once()

    def test_get_document_by_celex(self, client, mock_eurlex_response):
        """Test retrieving document by CELEX number."""
        with patch.object(client, '_execute_sparql') as mock_sparql:
            mock_sparql.return_value = mock_eurlex_response

            doc = client.get_document_by_celex("32016R0679")

            assert doc["celex"] == "32016R0679"
            assert doc["title"] == "General Data Protection Regulation"
            mock_sparql.assert_called_once()

    def test_get_document_by_celex_not_found(self, client):
        """Test error when document not found."""
        with patch.object(client, '_execute_sparql') as mock_sparql:
            mock_sparql.return_value = {"results": {"bindings": []}}

            with pytest.raises(ValueError, match="not found"):
                client.get_document_by_celex("INVALID")

    def test_download_document_success(self, client, mock_eurlex_document):
        """Test downloading document content."""
        with patch.object(client.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = mock_eurlex_document
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            content = client.download_document("32016R0679", format="xml")

            assert content == mock_eurlex_document
            assert b"<?xml" in content
            mock_get.assert_called_once()

    def test_download_document_different_formats(self, client):
        """Test downloading in different formats."""
        with patch.object(client.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"test content"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Test XML format
            client.download_document("32016R0679", format="xml")
            assert "fmx" in str(mock_get.call_args).upper() or "FMX" in str(mock_get.call_args)

            # Test PDF format
            client.download_document("32016R0679", format="pdf")

            # Test HTML format
            client.download_document("32016R0679", format="html")

    def test_download_document_to_file(self, client, mock_eurlex_document, tmp_path):
        """Test downloading document to file."""
        with patch.object(client, 'download_document') as mock_download:
            mock_download.return_value = mock_eurlex_document

            output_path = tmp_path / "test_doc.xml"
            result_path = client.download_document_to_file(
                "32016R0679",
                output_path,
                format="xml",
            )

            assert result_path == output_path
            assert output_path.exists()
            assert output_path.read_bytes() == mock_eurlex_document

    def test_get_gdpr_document(self, client, mock_eurlex_response):
        """Test getting GDPR document."""
        with patch.object(client, 'get_document_by_celex') as mock_get:
            mock_get.return_value = mock_eurlex_response["results"]["bindings"][0]

            doc = client.get_gdpr_document()

            mock_get.assert_called_once_with("32016R0679")

    def test_extract_celex_metadata(self, client):
        """Test extracting metadata from CELEX number."""
        with patch.object(client, 'get_document_by_celex') as mock_get:
            # Return flattened structure as get_document_by_celex actually does
            doc_data = {
                "work_uri": "http://publications.europa.eu/resource/cellar/...",
                "celex": "32016R0679",
                "title": "General Data Protection Regulation",
                "date": "2016-04-27",
                "type": "regulation",
            }
            mock_get.return_value = doc_data

            regulation = client.extract_celex_metadata("32016R0679")

            assert regulation.celex_id == "32016R0679"
            assert regulation.regulation_type == RegulationType.REGULATION
            assert regulation.name == "GDPR"

    def test_extract_regulation_name(self, client):
        """Test extracting short regulation name from title."""
        # Test GDPR
        assert client._extract_regulation_name(
            "Regulation 2016/679 on the protection of natural persons"
        ) == "GDPR"

        # Test AI Act
        assert client._extract_regulation_name(
            "Regulation on Artificial Intelligence"
        ) == "AI Act"

        # Test generic title
        name = client._extract_regulation_name("Some Random Regulation Title Here")
        assert len(name) > 0

    def test_context_manager(self):
        """Test client works as context manager."""
        with EURLexClient() as client:
            assert client.client is not None

        # Client should be closed after exiting context
        # Note: Can't directly test this without actually closing,
        # but we verify the __exit__ method exists
        assert hasattr(client, '__exit__')

    def test_close_method(self, client):
        """Test close method."""
        with patch.object(client.client, 'close') as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestEURLexIntegrationScenarios:
    """Integration-like tests for EUR-Lex client workflows."""

    @pytest.fixture
    def client(self):
        """Create EUR-Lex client."""
        return EURLexClient()

    def test_full_document_retrieval_workflow(self, client, mock_eurlex_response, mock_eurlex_document):
        """Test complete workflow: search -> metadata -> download."""
        with patch.object(client, '_execute_sparql') as mock_sparql, \
             patch.object(client.client, 'get') as mock_get:

            # Mock search
            mock_sparql.return_value = mock_eurlex_response

            # Mock download
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = mock_eurlex_document
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # 1. Search for documents
            docs = client.search_recent_documents(start_date=date(2024, 1, 1), limit=5)
            assert len(docs) > 0

            # 2. Get metadata for first document
            celex = docs[0]["celex"]
            doc_metadata = client.get_document_by_celex(celex)
            assert doc_metadata["celex"] == celex

            # 3. Download document content
            content = client.download_document(celex, format="xml")
            assert content is not None
            assert len(content) > 0
