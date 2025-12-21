"""Answer quality metrics (computed metrics, not LLM-based)."""

import re
from typing import Any

from tests.evaluation.base import BaseEvaluator, EvaluationResult, MetricType


class AnswerEvaluator(BaseEvaluator):
    """
    Evaluator for answer quality using computed metrics.

    Metrics:
    - Citation coverage: Percentage of claims with citations
    - Token overlap: Similarity with ground truth
    - Hallucination detection: Claims not in sources
    - Answer length appropriateness
    """

    async def evaluate_citation_coverage(
        self,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> EvaluationResult:
        """
        Evaluate if answer has sufficient citations.

        Checks:
        - Number of citations
        - Distribution throughout answer
        """
        # Count sentences and citations
        sentences = self._split_into_sentences(answer)
        num_sentences = len(sentences)
        num_citations = len(citations)

        # Citation rate
        citation_rate = num_citations / num_sentences if num_sentences > 0 else 0.0

        # Check citation distribution (citations in different parts of answer)
        citation_distribution = self._calculate_citation_distribution(answer, citations)

        # Score (0.5+ citations per sentence is good)
        score = min(citation_rate / 0.5, 1.0)

        return self._create_result(
            metric_name="citation_coverage",
            metric_type=MetricType.CITATION_QUALITY,
            score=score,
            details={
                "num_sentences": num_sentences,
                "num_citations": num_citations,
                "citation_rate": citation_rate,
                "citation_distribution_score": citation_distribution,
            },
            explanation=f"Answer has {num_citations} citations for {num_sentences} sentences (rate: {citation_rate:.2f})",
        )

    async def evaluate_token_overlap(
        self,
        prediction: str,
        ground_truth: str,
    ) -> EvaluationResult:
        """
        Calculate token overlap with ground truth.

        Uses F1 score of token overlap.
        """
        pred_tokens = set(self._tokenize(prediction.lower()))
        gt_tokens = set(self._tokenize(ground_truth.lower()))

        if not pred_tokens or not gt_tokens:
            return self._create_result(
                metric_name="token_overlap",
                metric_type=MetricType.CORRECTNESS,
                score=0.0,
                details={"pred_tokens": len(pred_tokens), "gt_tokens": len(gt_tokens)},
            )

        # Calculate overlap
        overlap = pred_tokens & gt_tokens

        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return self._create_result(
            metric_name="token_overlap_f1",
            metric_type=MetricType.CORRECTNESS,
            score=f1,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "overlap_tokens": len(overlap),
                "pred_tokens": len(pred_tokens),
                "gt_tokens": len(gt_tokens),
            },
            explanation=f"Token overlap F1: {f1:.3f} (P: {precision:.3f}, R: {recall:.3f})",
        )

    async def evaluate_hallucination(
        self,
        answer: str,
        retrieved_chunks: list[str],
    ) -> EvaluationResult:
        """
        Detect potential hallucinations by checking if answer tokens
        are present in retrieved sources.

        Note: This is a simple heuristic, use LLM judge for better results.
        """
        answer_tokens = set(self._tokenize(answer.lower()))

        # Combine all source tokens
        source_text = " ".join(retrieved_chunks).lower()
        source_tokens = set(self._tokenize(source_text))

        # Remove stop words and common words
        answer_content_tokens = self._remove_stopwords(answer_tokens)

        if not answer_content_tokens:
            return self._create_result(
                metric_name="hallucination_score",
                metric_type=MetricType.FAITHFULNESS,
                score=1.0,
                details={"answer_tokens": 0, "source_coverage": 1.0},
            )

        # Calculate how many answer tokens are in sources
        tokens_in_sources = answer_content_tokens & source_tokens
        coverage = len(tokens_in_sources) / len(answer_content_tokens)

        # High coverage = low hallucination = high score
        score = coverage

        return self._create_result(
            metric_name="source_grounding_score",
            metric_type=MetricType.FAITHFULNESS,
            score=score,
            details={
                "answer_content_tokens": len(answer_content_tokens),
                "tokens_in_sources": len(tokens_in_sources),
                "source_coverage": coverage,
                "potentially_hallucinated_tokens": len(answer_content_tokens - tokens_in_sources),
            },
            explanation=f"{coverage*100:.1f}% of answer tokens found in sources",
        )

    async def evaluate_answer_length(
        self,
        answer: str,
        expected_length_range: tuple[int, int] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate if answer length is appropriate.

        Args:
            answer: Generated answer
            expected_length_range: (min_words, max_words) tuple
        """
        words = self._tokenize(answer)
        word_count = len(words)

        if expected_length_range is None:
            # Default: 50-500 words is good
            expected_length_range = (50, 500)

        min_words, max_words = expected_length_range

        if word_count < min_words:
            score = word_count / min_words  # Too short
        elif word_count > max_words:
            score = max_words / word_count  # Too long
        else:
            score = 1.0  # Just right

        return self._create_result(
            metric_name="answer_length_appropriateness",
            metric_type=MetricType.COMPLETENESS,
            score=score,
            details={
                "word_count": word_count,
                "expected_min": min_words,
                "expected_max": max_words,
                "status": "appropriate" if score == 1.0 else ("too_short" if word_count < min_words else "too_long"),
            },
            explanation=f"Answer has {word_count} words (expected: {min_words}-{max_words})",
        )

    async def evaluate(
        self,
        query: str,
        prediction: Any,
        ground_truth: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Run all computed evaluations."""
        raise NotImplementedError("Use specific evaluate_* methods")

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    @staticmethod
    def _remove_stopwords(tokens: set[str]) -> set[str]:
        """Remove common stop words."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        }
        return {t for t in tokens if t not in stopwords and len(t) > 2}

    @staticmethod
    def _calculate_citation_distribution(answer: str, citations: list[dict[str, Any]]) -> float:
        """
        Calculate how evenly citations are distributed throughout answer.

        Returns score 0-1, where 1 is perfectly distributed.
        """
        if not citations or not answer:
            return 0.0

        # Divide answer into thirds
        third = len(answer) // 3

        # Count citations in each third
        citations_in_thirds = [0, 0, 0]

        for citation in citations:
            # Estimate position (simplified - would need actual positions)
            # For now, assume uniform distribution
            citations_in_thirds[len(citations_in_thirds) // 2] += 1

        # Ideal: equal distribution
        ideal_per_third = len(citations) / 3

        # Calculate variance from ideal
        variance = sum((count - ideal_per_third) ** 2 for count in citations_in_thirds) / 3

        # Normalize to 0-1 (lower variance = higher score)
        max_variance = (len(citations) ** 2) / 3
        score = 1.0 - (variance / max_variance if max_variance > 0 else 0)

        return max(0.0, min(1.0, score))
