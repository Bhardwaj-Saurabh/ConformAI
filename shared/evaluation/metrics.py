"""
Evaluation Metrics for ConformAI

Custom scoring functions for evaluating RAG system performance.
"""

import re
from typing import Any

from shared.utils import get_logger

logger = get_logger(__name__)


def exact_match_score(expected: str, actual: str) -> float:
    """
    Exact match scoring (1.0 if identical, 0.0 otherwise).

    Args:
        expected: Expected output
        actual: Actual model output

    Returns:
        1.0 if match, 0.0 otherwise
    """
    return 1.0 if expected.strip() == actual.strip() else 0.0


def contains_keywords_score(expected: str, actual: str) -> float:
    """
    Score based on presence of key terms from expected output.

    Args:
        expected: Expected output
        actual: Actual model output

    Returns:
        Proportion of keywords present (0.0 to 1.0)
    """
    # Extract key terms (words longer than 3 characters)
    expected_words = set(
        word.lower() for word in re.findall(r"\b\w{4,}\b", expected)
    )

    if not expected_words:
        return 1.0  # No keywords to check

    actual_words = set(word.lower() for word in re.findall(r"\b\w{4,}\b", actual))

    # Calculate overlap
    overlap = len(expected_words & actual_words)
    total = len(expected_words)

    return overlap / total if total > 0 else 0.0


def citation_presence_score(
    expected: str, actual: str, citations: list[dict] | None = None
) -> float:
    """
    Score based on presence of citations in the response.

    Args:
        expected: Expected output (not used)
        actual: Actual model output
        citations: List of citation objects

    Returns:
        1.0 if citations present, 0.0 otherwise
    """
    if citations and len(citations) > 0:
        return 1.0
    return 0.0


def regulation_mention_score(expected: str, actual: str) -> float:
    """
    Score based on correct regulation/article mentions.

    Args:
        expected: Expected output
        actual: Actual model output

    Returns:
        Score from 0.0 to 1.0
    """
    # Extract regulation mentions (e.g., "Article 5", "GDPR", "EU AI Act")
    regulation_patterns = [
        r"Article \d+",
        r"Annex [IVX]+",
        r"GDPR",
        r"EU AI Act",
        r"Regulation \(EU\) \d+/\d+",
    ]

    expected_mentions = set()
    actual_mentions = set()

    for pattern in regulation_patterns:
        expected_mentions.update(re.findall(pattern, expected, re.IGNORECASE))
        actual_mentions.update(re.findall(pattern, actual, re.IGNORECASE))

    # Normalize to lowercase
    expected_mentions = {m.lower() for m in expected_mentions}
    actual_mentions = {m.lower() for m in actual_mentions}

    if not expected_mentions:
        return 1.0  # No mentions to check

    # Calculate overlap
    overlap = len(expected_mentions & actual_mentions)
    total = len(expected_mentions)

    return overlap / total if total > 0 else 0.0


def answer_length_score(expected: str, actual: str, tolerance: float = 0.5) -> float:
    """
    Score based on answer length similarity.

    Args:
        expected: Expected output
        actual: Actual model output
        tolerance: Acceptable deviation ratio (0.5 = ±50%)

    Returns:
        Score from 0.0 to 1.0
    """
    expected_len = len(expected.split())
    actual_len = len(actual.split())

    if expected_len == 0:
        return 1.0

    ratio = actual_len / expected_len

    # Score based on how close to expected length
    if 1 - tolerance <= ratio <= 1 + tolerance:
        # Within tolerance - perfect score
        return 1.0
    elif ratio < 1 - tolerance:
        # Too short
        return ratio / (1 - tolerance)
    else:
        # Too long
        return (1 + tolerance) / ratio


def semantic_similarity_score(expected: str, actual: str) -> float:
    """
    Semantic similarity using simple overlap (placeholder for embedding-based).

    Args:
        expected: Expected output
        actual: Actual model output

    Returns:
        Score from 0.0 to 1.0

    Note:
        This is a simplified version. For production, use embedding-based similarity
        with models like sentence-transformers.
    """
    # Simple word overlap as proxy for semantic similarity
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())

    if not expected_words:
        return 1.0

    intersection = len(expected_words & actual_words)
    union = len(expected_words | actual_words)

    return intersection / union if union > 0 else 0.0


def comprehensive_score(
    expected: str,
    actual: str,
    citations: list[dict] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute comprehensive evaluation with multiple metrics.

    Args:
        expected: Expected output
        actual: Actual model output
        citations: Optional list of citations
        weights: Optional weights for combining metrics

    Returns:
        Dict with individual scores and weighted average
    """
    if weights is None:
        weights = {
            "keywords": 0.3,
            "regulation": 0.25,
            "semantic": 0.25,
            "length": 0.1,
            "citations": 0.1,
        }

    scores = {
        "keywords": contains_keywords_score(expected, actual),
        "regulation": regulation_mention_score(expected, actual),
        "semantic": semantic_similarity_score(expected, actual),
        "length": answer_length_score(expected, actual),
        "citations": citation_presence_score(expected, actual, citations),
    }

    # Calculate weighted average
    weighted_avg = sum(scores[k] * weights.get(k, 0) for k in scores)

    scores["weighted_average"] = weighted_avg

    return scores


# Opik-compatible scoring functions
class OpikKeywordsScorer:
    """Opik-compatible keywords scorer."""

    def __init__(self):
        self.name = "keywords_score"

    def __call__(self, output: dict[str, Any]) -> float:
        """Score based on keyword presence."""
        expected = output.get("expected_output", "")
        actual = output.get("output", "")
        return contains_keywords_score(expected, actual)


class OpikRegulationScorer:
    """Opik-compatible regulation mention scorer."""

    def __init__(self):
        self.name = "regulation_score"

    def __call__(self, output: dict[str, Any]) -> float:
        """Score based on regulation mentions."""
        expected = output.get("expected_output", "")
        actual = output.get("output", "")
        return regulation_mention_score(expected, actual)


class OpikCitationScorer:
    """Opik-compatible citation presence scorer."""

    def __init__(self):
        self.name = "citation_score"

    def __call__(self, output: dict[str, Any]) -> float:
        """Score based on citation presence."""
        citations = output.get("citations", [])
        return 1.0 if citations else 0.0


def get_default_scorers() -> list:
    """Get default Opik-compatible scoring functions."""
    return [
        OpikKeywordsScorer(),
        OpikRegulationScorer(),
        OpikCitationScorer(),
    ]
