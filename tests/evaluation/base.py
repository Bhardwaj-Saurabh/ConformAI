"""Base classes for evaluation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    RETRIEVAL = "retrieval"
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    TOOL_USE = "tool_use"
    PERFORMANCE = "performance"
    CITATION_QUALITY = "citation_quality"


@dataclass
class EvaluationResult:
    """Result from an evaluation metric."""

    metric_name: str
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    passed: bool  # Whether it meets threshold
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    explanation: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": self.details,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EvaluationMetrics:
    """Collection of evaluation results."""

    test_case_id: str
    query: str
    results: list[EvaluationResult] = field(default_factory=list)
    overall_score: float = 0.0
    passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.results.append(result)
        self._update_overall_score()

    def _update_overall_score(self) -> None:
        """Calculate overall score from all results."""
        if not self.results:
            self.overall_score = 0.0
            self.passed = False
            return

        # Weighted average (can be customized)
        total_score = sum(r.score for r in self.results)
        self.overall_score = total_score / len(self.results)

        # All metrics must pass
        self.passed = all(r.passed for r in self.results)

    def get_failed_metrics(self) -> list[EvaluationResult]:
        """Get all failed metrics."""
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "query": self.query,
            "overall_score": self.overall_score,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, threshold: float = 0.7):
        """
        Initialize evaluator.

        Args:
            threshold: Minimum score to pass (0.0 to 1.0)
        """
        self.threshold = threshold

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        prediction: Any,
        ground_truth: Any | None = None,
        context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a prediction.

        Args:
            query: Input query
            prediction: Model prediction/output
            ground_truth: Ground truth answer (if available)
            context: Additional context (retrieved chunks, etc.)

        Returns:
            Evaluation result
        """
        pass

    def _create_result(
        self,
        metric_name: str,
        metric_type: MetricType,
        score: float,
        details: dict[str, Any] | None = None,
        explanation: str | None = None,
    ) -> EvaluationResult:
        """Helper to create evaluation result."""
        return EvaluationResult(
            metric_name=metric_name,
            metric_type=metric_type,
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            details=details or {},
            explanation=explanation,
        )
