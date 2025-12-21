"""
Pytest Configuration and Shared Fixtures

Provides reusable fixtures for unit, integration, and e2e tests.
"""

import asyncio
import os
import sys
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "rag-service" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "data-pipeline" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from shared.config import Settings, get_settings
from shared.models import (
    AIDomain,
    Chunk,
    ChunkMetadata,
    LegalDocument,
    Regulation,
    RegulationType,
    RiskCategory,
)

# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest."""
    # Set test environment
    os.environ["ENVIRONMENT"] = "development"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Mock API keys for tests that don't actually call APIs
    if "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "sk-test-key"


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Auto-mark tests based on path
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# ============================================================================
# Event Loop Fixture (for async tests)
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Get test settings."""
    return get_settings()


@pytest.fixture
def mock_settings() -> Mock:
    """Mock settings for isolated tests."""
    settings = Mock(spec=Settings)
    settings.anthropic_api_key = "sk-ant-test-key"
    settings.openai_api_key = "sk-test-key"
    settings.qdrant_url = "http://localhost:6333"
    settings.qdrant_collection_name = "test_collection"
    settings.embedding_dimension = 1024
    settings.llm_model = "claude-3-5-sonnet-20241022"
    settings.embedding_model = "text-embedding-3-large"
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    settings.retrieval_top_k = 10
    settings.log_level = "DEBUG"
    return settings


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_regulation() -> Regulation:
    """Create sample regulation."""
    return Regulation(
        celex_id="32016R0679",
        name="GDPR",
        full_title="General Data Protection Regulation",
        regulation_type=RegulationType.REGULATION,
        adoption_date=date(2016, 4, 27),
        effective_date=date(2018, 5, 25),
        url="https://eur-lex.europa.eu/eli/reg/2016/679/oj",
        version="1.0",
        is_active=True,
    )


@pytest.fixture
def sample_chunk_metadata() -> ChunkMetadata:
    """Create sample chunk metadata."""
    return ChunkMetadata(
        regulation_name="GDPR",
        celex_id="32016R0679",
        regulation_type=RegulationType.REGULATION,
        chapter_number="IV",
        chapter_title="Controller and Processor",
        article_number="22",
        article_title="Automated individual decision-making",
        paragraph_index=1,
        chunk_index=0,
        total_chunks=1,
        effective_date=date(2018, 5, 25),
        version="1.0",
        domains=[AIDomain.GENERAL],
        risk_category=RiskCategory.HIGH_RISK,
        source_url="https://eur-lex.europa.eu/eli/reg/2016/679/oj",
        last_updated=datetime.now(),
    )


@pytest.fixture
def sample_chunk(sample_chunk_metadata) -> Chunk:
    """Create sample chunk without embedding."""
    return Chunk(
        text="The data subject shall have the right not to be subject to a decision based solely on automated processing.",
        metadata=sample_chunk_metadata,
        embedding=None,
    )


@pytest.fixture
def sample_chunk_with_embedding(sample_chunk_metadata) -> Chunk:
    """Create sample chunk with mock embedding."""
    import random
    return Chunk(
        text="The data subject shall have the right not to be subject to a decision based solely on automated processing.",
        metadata=sample_chunk_metadata,
        embedding=[random.random() for _ in range(1024)],
    )


@pytest.fixture
def sample_chunks(sample_chunk_metadata) -> list[Chunk]:
    """Create list of sample chunks."""
    chunks = []
    for i in range(5):
        metadata = sample_chunk_metadata.model_copy()
        metadata.chunk_index = i
        metadata.total_chunks = 5
        chunk = Chunk(
            text=f"Sample chunk text {i}. This is a test chunk for unit testing.",
            metadata=metadata,
            embedding=None,
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_document(sample_regulation) -> LegalDocument:
    """Create sample document."""
    return LegalDocument(
        regulation=sample_regulation,
        chapters=[],
        annexes={},
        raw_text="<root><article>Sample article content</article></root>",
    )


# ============================================================================
# Mock API Fixtures
# ============================================================================

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client."""
    with patch("anthropic.Anthropic") as mock:
        client = Mock()

        # Mock messages.create
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response.")]
        mock_response.usage = Mock(
            input_tokens=100,
            output_tokens=50,
        )

        client.messages.create = Mock(return_value=mock_response)
        mock.return_value = client

        yield client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client with dynamic batch size support."""
    with patch("openai.OpenAI") as mock:
        client = Mock()

        # Mock embeddings.create with side_effect to handle variable batch sizes
        def create_embeddings(**kwargs):
            input_data = kwargs.get("input", [])
            # Handle both single string and list of strings
            batch_size = len(input_data) if isinstance(input_data, list) else 1

            # Create mock response with correct number of embeddings
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1024) for _ in range(batch_size)]
            return mock_response

        client.embeddings.create = Mock(side_effect=create_embeddings)
        mock.return_value = client

        yield client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client (API v1.8+)."""
    with patch("qdrant_client.QdrantClient") as mock:
        client = Mock()

        # Mock common methods for Qdrant client API v1.8+
        client.get_collections = Mock(return_value=Mock(collections=[]))
        client.create_collection = Mock()
        client.upsert = Mock()

        # Mock query_points (new API) - returns QueryResponse with points attribute
        mock_query_response = Mock()
        mock_query_response.points = []
        client.query_points = Mock(return_value=mock_query_response)

        client.get_collection = Mock(return_value=Mock(points_count=0))

        mock.return_value = client
        yield client


# ============================================================================
# EUR-Lex Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_eurlex_response():
    """Mock EUR-Lex SPARQL response."""
    return {
        "results": {
            "bindings": [
                {
                    "work": {"value": "http://publications.europa.eu/resource/cellar/..."},
                    "celex": {"value": "32016R0679"},
                    "title": {"value": "General Data Protection Regulation"},
                    "date": {"value": "2016-04-27"},
                    "type": {"value": "regulation"},
                }
            ]
        }
    }


@pytest.fixture
def mock_eurlex_document():
    """Mock EUR-Lex XML document."""
    return b"""<?xml version="1.0" encoding="UTF-8"?>
    <root>
        <chapter number="IV" title="Controller and Processor">
            <article number="22" title="Automated individual decision-making">
                <paragraph>
                    The data subject shall have the right not to be subject to a decision
                    based solely on automated processing.
                </paragraph>
            </article>
        </chapter>
    </root>
    """


# ============================================================================
# Database Fixtures (for integration tests)
# ============================================================================

@pytest.fixture(scope="session")
def qdrant_test_client(test_settings):
    """Real Qdrant client for integration tests."""
    pytest.importorskip("qdrant_client")

    try:
        client = QdrantClient(url=test_settings.qdrant_url)
        # Test connection
        client.get_collections()
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"Qdrant not available: {e}")


@pytest.fixture
def qdrant_test_collection(qdrant_test_client):
    """Create test collection in Qdrant."""
    collection_name = "test_collection"

    # Delete if exists
    try:
        qdrant_test_client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection
    qdrant_test_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    yield collection_name

    # Cleanup
    try:
        qdrant_test_client.delete_collection(collection_name)
    except Exception:
        pass


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "processed" / "chunks").mkdir(parents=True)
    (data_dir / "embeddings").mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_xml_file(temp_data_dir):
    """Create sample XML file."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <root>
        <chapter number="IV" title="Test Chapter">
            <article number="22" title="Test Article">
                <paragraph>Test paragraph content.</paragraph>
            </article>
        </chapter>
    </root>
    """

    xml_file = temp_data_dir / "raw" / "test.xml"
    xml_file.write_text(xml_content)
    return xml_file


# ============================================================================
# Logger Fixtures
# ============================================================================

@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch("shared.utils.logger.get_logger") as mock:
        logger = Mock()
        logger.info = Mock()
        logger.debug = Mock()
        logger.warning = Mock()
        logger.error = Mock()
        logger.log_performance = Mock()
        logger.log_audit = Mock()
        logger.log_error_with_context = Mock()

        mock.return_value = logger
        yield logger


# ============================================================================
# HTTP Client Fixtures
# ============================================================================

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client."""
    with patch("httpx.Client") as mock:
        client = Mock()

        # Mock get/post methods
        response = Mock()
        response.status_code = 200
        response.json = Mock(return_value={})
        response.content = b"test content"
        response.raise_for_status = Mock()

        client.get = Mock(return_value=response)
        client.post = Mock(return_value=response)

        mock.return_value = client
        yield client


# ============================================================================
# Async Fixtures
# ============================================================================

@pytest.fixture
async def async_mock_anthropic():
    """Async mock for Anthropic client."""
    mock = AsyncMock()
    mock.messages.create = AsyncMock(
        return_value=Mock(
            content=[Mock(text="Test response")],
            usage=Mock(input_tokens=10, output_tokens=20),
        )
    )
    return mock


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_chunks():
    """Factory fixture for generating test chunks."""
    def _generate(count: int = 5, with_embeddings: bool = False):
        import random
        chunks = []
        for i in range(count):
            metadata = ChunkMetadata(
                regulation_name="Test Regulation",
                celex_id="TEST001",
                regulation_type=RegulationType.REGULATION,
                article_number=str(i + 1),
                article_title=f"Article {i + 1}",
                chunk_index=i,
                total_chunks=count,
            )
            chunk = Chunk(
                text=f"Test chunk {i} with some meaningful content for testing purposes.",
                metadata=metadata,
                embedding=[random.random() for _ in range(1024)] if with_embeddings else None,
            )
            chunks.append(chunk)
        return chunks

    return _generate


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files(temp_data_dir):
    """Auto-cleanup test files after each test."""
    yield
    # Cleanup happens automatically with tmp_path
