"""
Unit Tests for Shared Models

Tests Pydantic models for data validation and serialization.
"""

from datetime import date

from shared.models import (
    AIDomain,
    Chunk,
    ChunkMetadata,
    LegalDocument,
    Regulation,
    RegulationType,
    RiskCategory,
)


class TestRegulation:
    """Tests for Regulation model."""

    def test_regulation_creation(self):
        """Test creating a regulation with valid data."""
        reg = Regulation(
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

        assert reg.celex_id == "32016R0679"
        assert reg.name == "GDPR"
        assert reg.regulation_type == RegulationType.REGULATION
        assert reg.is_active is True

    def test_regulation_serialization(self):
        """Test regulation can be serialized to dict."""
        reg = Regulation(
            celex_id="32016R0679",
            name="GDPR",
            full_title="General Data Protection Regulation",
            regulation_type=RegulationType.REGULATION,
        )

        reg_dict = reg.model_dump()
        assert reg_dict["celex_id"] == "32016R0679"
        assert reg_dict["name"] == "GDPR"

    def test_regulation_from_dict(self):
        """Test creating regulation from dictionary."""
        data = {
            "celex_id": "32016R0679",
            "name": "GDPR",
            "full_title": "General Data Protection Regulation",
            "regulation_type": "regulation",
        }

        reg = Regulation(**data)
        assert reg.celex_id == "32016R0679"


class TestChunkMetadata:
    """Tests for ChunkMetadata model."""

    def test_chunk_metadata_creation(self):
        """Test creating chunk metadata."""
        metadata = ChunkMetadata(
            regulation_name="GDPR",
            celex_id="32016R0679",
            regulation_type=RegulationType.REGULATION,
            article_number="22",
            article_title="Automated decision-making",
            chunk_index=0,
            total_chunks=3,
        )

        assert metadata.regulation_name == "GDPR"
        assert metadata.article_number == "22"
        assert metadata.chunk_index == 0

    def test_chunk_metadata_with_domains(self):
        """Test chunk metadata with AI use case domains."""
        metadata = ChunkMetadata(
            regulation_name="AI Act",
            celex_id="TEST001",
            regulation_type=RegulationType.REGULATION,
            article_number="10",
            chunk_index=0,
            total_chunks=1,
            domains=[AIDomain.BIOMETRICS, AIDomain.RECRUITMENT],
        )

        assert len(metadata.domains) == 2
        assert AIDomain.BIOMETRICS in metadata.domains

    def test_chunk_metadata_with_risk_category(self):
        """Test chunk metadata with risk category."""
        metadata = ChunkMetadata(
            regulation_name="AI Act",
            celex_id="TEST001",
            regulation_type=RegulationType.REGULATION,
            article_number="10",
            chunk_index=0,
            total_chunks=1,
            risk_category=RiskCategory.HIGH_RISK,
        )

        assert metadata.risk_category == RiskCategory.HIGH_RISK


class TestChunk:
    """Tests for Chunk model."""

    def test_chunk_creation_without_embedding(self, sample_chunk_metadata):
        """Test creating chunk without embedding."""
        chunk = Chunk(
            text="Sample chunk text.",
            metadata=sample_chunk_metadata,
            embedding=None,
        )

        assert chunk.text == "Sample chunk text."
        assert chunk.embedding is None
        assert chunk.metadata.regulation_name == "GDPR"

    def test_chunk_creation_with_embedding(self, sample_chunk_metadata):
        """Test creating chunk with embedding."""
        embedding = [0.1, 0.2, 0.3] * 100  # 300 dimensions

        chunk = Chunk(
            text="Sample chunk text.",
            metadata=sample_chunk_metadata,
            embedding=embedding,
        )

        assert chunk.embedding is not None
        assert len(chunk.embedding) == 300

    def test_chunk_text_length_property(self, sample_chunk):
        """Test chunk text length is accessible."""
        assert len(sample_chunk.text) > 0

    def test_chunk_serialization(self, sample_chunk_with_embedding):
        """Test chunk serialization."""
        chunk_dict = sample_chunk_with_embedding.model_dump()

        assert "text" in chunk_dict
        assert "metadata" in chunk_dict
        assert "embedding" in chunk_dict
        assert isinstance(chunk_dict["metadata"], dict)


class TestLegalDocument:
    """Tests for LegalDocument model."""

    def test_document_creation(self, sample_regulation):
        """Test creating a legal document."""
        doc = LegalDocument(
            regulation=sample_regulation,
            chapters=[],
            annexes={},
            raw_text="<root>Test content</root>",
        )

        assert doc.regulation.name == "GDPR"
        assert "Test content" in doc.raw_text
        assert len(doc.chapters) == 0

    def test_document_with_chapters(self, sample_regulation):
        """Test document with chapters."""
        from shared.models import Chapter, Article

        article = Article(
            number="22",
            title="Automated decision-making",
            content="Test article content",
        )

        chapter = Chapter(
            number="III",
            title="Rights of the data subject",
            articles=[article],
        )

        doc = LegalDocument(
            regulation=sample_regulation,
            chapters=[chapter],
            raw_text="<root>Test content</root>",
        )

        assert len(doc.chapters) == 1
        assert doc.chapters[0].number == "III"
        assert len(doc.chapters[0].articles) == 1


class TestEnums:
    """Tests for enum types."""

    def test_regulation_type_enum(self):
        """Test RegulationType enum."""
        assert RegulationType.REGULATION.value == "regulation"
        assert RegulationType.DIRECTIVE.value == "directive"
        assert RegulationType.DECISION.value == "decision"

    def test_risk_category_enum(self):
        """Test RiskCategory enum."""
        assert RiskCategory.PROHIBITED.value == "prohibited"
        assert RiskCategory.HIGH_RISK.value == "high"
        assert RiskCategory.LIMITED_RISK.value == "limited"
        assert RiskCategory.MINIMAL_RISK.value == "minimal"

    def test_ai_domain_enum(self):
        """Test AIDomain enum."""
        assert AIDomain.BIOMETRICS in AIDomain
        assert AIDomain.RECRUITMENT in AIDomain
        assert AIDomain.EDUCATION in AIDomain
        assert AIDomain.HEALTHCARE in AIDomain
