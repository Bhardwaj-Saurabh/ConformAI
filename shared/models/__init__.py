"""Shared data models for ConformAI."""

from .legal_document import (
    AIDomain,
    Article,
    Chapter,
    Chunk,
    ChunkMetadata,
    LegalDocument,
    QueryClassification,
    Regulation,
    RiskCategory,
)

__all__ = [
    "LegalDocument",
    "Regulation",
    "Chapter",
    "Article",
    "Chunk",
    "ChunkMetadata",
    "QueryClassification",
    "AIDomain",
    "RiskCategory",
]
