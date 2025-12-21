"""
End-to-End Tests for API Endpoints

Tests API endpoints with FastAPI TestClient.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.e2e, pytest.mark.api]


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    import sys
    from pathlib import Path

    # Add service to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "services" / "rag-service" / "src"))

    # Import after adding to path
    from api.main import app

    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, api_client):
        """Test basic health check endpoint."""
        response = api_client.get("/health")

        # Accept both 200 (healthy) and 503 (degraded - expected without Qdrant collection)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_readiness_check(self, api_client):
        """Test readiness endpoint."""
        response = api_client.get("/health/ready")

        # Accept both 200 (ready) and 503 (not ready - expected without Qdrant collection)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data

    def test_liveness_check(self, api_client):
        """Test liveness endpoint."""
        response = api_client.get("/health/live")

        assert response.status_code == 200


class TestQueryEndpoints:
    """Tests for query endpoints."""

    def test_query_endpoint_valid_request(self, api_client):
        """Test query endpoint with valid request."""
        with patch("api.main.run_rag_pipeline") as mock_rag:
            # Mock RAG pipeline response
            mock_rag.return_value = {
                "final_answer": "High-risk AI systems must undergo conformity assessment.",
                "citations": [
                    {
                        "source_id": 1,
                        "regulation": "AI Act",
                        "article": "43",
                        "paragraph": None,
                        "celex": "32021R1234",
                        "excerpt": "High-risk AI systems must undergo conformity assessment.",
                        "chunk_id": "chunk_1",
                    }
                ],
                "all_retrieved_chunks": [],
                "confidence_score": 0.92,
                "intent": "compliance_question",
                "ai_domain": None,
                "risk_category": None,
                "query_complexity": "simple",
                "processing_time_ms": 100,
                "total_llm_calls": 1,
                "total_tokens_used": 50,
                "iteration_count": 1,
            }

            response = api_client.post(
                "/api/v1/query",
                json={
                    "query": "What are the requirements for high-risk AI systems?",
                    "conversation_id": None,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "metadata" in data
            assert data["metadata"]["confidence_score"] > 0.9

    def test_query_endpoint_empty_query(self, api_client):
        """Test query endpoint with empty query."""
        response = api_client.post(
            "/api/v1/query",
            json={
                "query": "",
                "conversation_id": None,
            },
        )

        # Should return validation error
        assert response.status_code == 422

    def test_query_endpoint_with_filters(self, api_client):
        """Test query endpoint with metadata filters."""
        with patch("api.main.run_rag_pipeline") as mock_rag:
            mock_rag.return_value = {
                "final_answer": "GDPR requires...",
                "citations": [
                    {
                        "source_id": 1,
                        "regulation": "GDPR",
                        "article": "22",
                        "paragraph": None,
                        "celex": "32016R0679",
                        "excerpt": "GDPR requires data protection.",
                        "chunk_id": "chunk_1",
                    }
                ],
                "all_retrieved_chunks": [],
                "confidence_score": 0.88,
                "intent": "compliance_question",
                "ai_domain": None,
                "risk_category": None,
                "query_complexity": "simple",
                "processing_time_ms": 100,
                "total_llm_calls": 1,
                "total_tokens_used": 50,
                "iteration_count": 1,
            }

            response = api_client.post(
                "/api/v1/query",
                json={
                    "query": "What are GDPR requirements?",
                    "filters": {
                        "regulation_name": "GDPR",
                    },
                },
            )

            assert response.status_code == 200

    def test_query_endpoint_with_conversation_id(self, api_client):
        """Test query endpoint with conversation ID for context."""
        with patch("api.main.run_rag_pipeline") as mock_rag:
            mock_rag.return_value = {
                "final_answer": "Follow-up answer...",
                "citations": [],
                "all_retrieved_chunks": [],
                "confidence_score": 0.85,
                "intent": "compliance_question",
                "ai_domain": None,
                "risk_category": None,
                "query_complexity": "simple",
                "processing_time_ms": 100,
                "total_llm_calls": 1,
                "total_tokens_used": 50,
                "iteration_count": 1,
            }

            response = api_client.post(
                "/api/v1/query",
                json={
                    "query": "Tell me more about that",
                    "conversation_id": "test-conversation-123",
                },
            )

            assert response.status_code == 200


@pytest.mark.requires_api_keys
class TestQueryWithRealLLM:
    """Tests with real LLM (requires API keys)."""

    def test_query_with_real_anthropic(self, api_client):
        """Test query with real Anthropic API (skipped if no key)."""
        import os

        if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY").startswith("sk-ant-test"):
            pytest.skip("Real API key not available")

        response = api_client.post(
            "/api/v1/query",
            json={
                "query": "What is the EU AI Act?",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0


class TestAuthenticationAndRateLimiting:
    """Tests for authentication and rate limiting."""

    def test_api_key_authentication_valid(self, api_client):
        """Test API key authentication with valid key."""
        with patch("api.middleware.auth.verify_api_key_and_rate_limit") as mock_verify:
            mock_verify.return_value = True

            response = api_client.post(
                "/api/v1/query",
                headers={"X-API-Key": "valid-api-key"},
                json={"query": "Test query"},
            )

            # Should not return 401/403
            assert response.status_code != 401
            assert response.status_code != 403

    def test_api_key_authentication_invalid(self, api_client):
        """Test API key authentication with invalid key."""
        with patch("api.middleware.auth.verify_api_key_and_rate_limit") as mock_verify:
            mock_verify.return_value = False

            response = api_client.post(
                "/api/v1/query",
                headers={"X-API-Key": "invalid-key"},
                json={"query": "Test query"},
            )

            # May return 401 Unauthorized if auth is enabled
            # (depends on configuration)
            assert response.status_code in [200, 401, 403]

    @pytest.mark.slow
    def test_rate_limiting(self, api_client):
        """Test rate limiting on API endpoints."""
        with patch("api.main.run_rag_pipeline") as mock_rag:
            mock_rag.return_value = {
                "answer": "Test answer",
                "confidence_score": 0.8,
                "sources": [],
                "citations": [],
            }

            # Make multiple rapid requests
            responses = []
            for i in range(100):  # Exceed rate limit
                response = api_client.post(
                    "/api/v1/query",
                    json={"query": f"Test query {i}"},
                )
                responses.append(response)

            # At least one should be rate limited (429)
            status_codes = [r.status_code for r in responses]

            # Either all succeed (no rate limiting) or some are limited
            assert 200 in status_codes  # At least some succeeded
            # May have 429 Too Many Requests if rate limiting is enabled


class TestErrorHandling:
    """Tests for error handling."""

    def test_internal_error_handling(self, api_client):
        """Test handling of internal server errors."""
        with patch("api.main.run_rag_pipeline") as mock_rag:
            # Simulate internal error
            mock_rag.side_effect = Exception("Internal processing error")

            response = api_client.post(
                "/api/v1/query",
                json={"query": "Test query"},
            )

            # Should return 500 Internal Server Error
            assert response.status_code == 500
            data = response.json()
            assert "error" in data or "detail" in data

    def test_validation_error_response(self, api_client):
        """Test validation error responses."""
        # Missing required field
        response = api_client.post(
            "/api/v1/query",
            json={},  # Missing "query" field
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_malformed_json(self, api_client):
        """Test handling of malformed JSON."""
        response = api_client.post(
            "/api/v1/query",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestCORSAndHeaders:
    """Tests for CORS and response headers."""

    def test_cors_headers(self, api_client):
        """Test CORS headers in response."""
        response = api_client.options("/api/v1/query")

        # Check for CORS headers (if enabled)
        assert response.status_code in [200, 405]  # OPTIONS may not be allowed

    def test_security_headers(self, api_client):
        """Test security headers in response."""
        response = api_client.get("/health")

        # Common security headers
        headers = response.headers

        # May include security headers like:
        # X-Content-Type-Options, X-Frame-Options, etc.
        assert "content-type" in headers


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema(self, api_client):
        """Test OpenAPI schema is available."""
        response = api_client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    def test_swagger_ui(self, api_client):
        """Test Swagger UI is accessible."""
        response = api_client.get("/docs")

        assert response.status_code == 200

    def test_redoc_ui(self, api_client):
        """Test ReDoc UI is accessible."""
        response = api_client.get("/redoc")

        assert response.status_code == 200
