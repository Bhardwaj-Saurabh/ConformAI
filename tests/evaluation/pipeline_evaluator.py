"""End-to-end pipeline evaluator combining all metrics."""

import asyncio
from dataclasses import dataclass
from typing import Any

from shared.utils.logger import get_logger
from tests.evaluation.answer_metrics import AnswerEvaluator
from tests.evaluation.base import EvaluationMetrics
from tests.evaluation.llm_judge import LLMJudge
from tests.evaluation.retrieval_metrics import RetrievalEvaluator

logger = get_logger(__name__)


@dataclass
class TestCase:
    """A test case for RAG evaluation."""

    id: str
    query: str
    ground_truth_answer: str | None = None
    relevant_chunk_ids: list[str] | None = None
    expected_aspects: list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class PipelineOutput:
    """Output from RAG pipeline."""

    query: str
    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[str]
    citations: list[dict[str, Any]]
    metadata: dict[str, Any]


class PipelineEvaluator:
    """
    Comprehensive end-to-end pipeline evaluator.

    Evaluates:
    1. Retrieval quality (if ground truth available)
    2. Answer faithfulness (LLM judge)
    3. Answer relevance (LLM judge)
    4. Answer correctness (LLM judge)
    5. Answer completeness (LLM judge)
    6. Citation quality (LLM judge + computed)
    7. Performance metrics
    """

    def __init__(
        self,
        retrieval_threshold: float = 0.7,
        answer_threshold: float = 0.7,
        judge_model: str = "claude-3-5-sonnet-20241022",
    ):
        """
        Initialize pipeline evaluator.

        Args:
            retrieval_threshold: Threshold for retrieval metrics
            answer_threshold: Threshold for answer quality metrics
            judge_model: Model to use as LLM judge
        """
        self.retrieval_evaluator = RetrievalEvaluator(threshold=retrieval_threshold)
        self.answer_evaluator = AnswerEvaluator(threshold=answer_threshold)
        self.llm_judge = LLMJudge(threshold=answer_threshold, judge_model=judge_model)

    async def evaluate_pipeline(
        self,
        test_case: TestCase,
        pipeline_output: PipelineOutput,
    ) -> EvaluationMetrics:
        """
        Evaluate complete pipeline output.

        Args:
            test_case: Test case with query and ground truth
            pipeline_output: Output from RAG pipeline

        Returns:
            Complete evaluation metrics
        """
        metrics = EvaluationMetrics(
            test_case_id=test_case.id,
            query=test_case.query,
            metadata=pipeline_output.metadata,
        )

        logger.info(f"Evaluating pipeline for test case: {test_case.id}")

        # Run all evaluations in parallel
        tasks = []

        # 1. Retrieval evaluation (if ground truth available)
        if test_case.relevant_chunk_ids:
            tasks.append(
                self._evaluate_retrieval(
                    test_case.query,
                    pipeline_output.retrieved_chunk_ids,
                    test_case.relevant_chunk_ids,
                )
            )

        # 2. LLM Judge evaluations
        tasks.extend([
            self.llm_judge.evaluate_faithfulness(
                query=test_case.query,
                answer=pipeline_output.answer,
                retrieved_chunks=pipeline_output.retrieved_chunks,
            ),
            self.llm_judge.evaluate_relevance(
                query=test_case.query,
                answer=pipeline_output.answer,
            ),
            self.llm_judge.evaluate_completeness(
                query=test_case.query,
                answer=pipeline_output.answer,
                expected_aspects=test_case.expected_aspects,
            ),
            self.llm_judge.evaluate_citation_quality(
                answer=pipeline_output.answer,
                citations=pipeline_output.citations,
                retrieved_chunks=pipeline_output.retrieved_chunks,
            ),
        ])

        # 3. Correctness (if ground truth available)
        if test_case.ground_truth_answer:
            tasks.append(
                self.llm_judge.evaluate_correctness(
                    query=test_case.query,
                    answer=pipeline_output.answer,
                    ground_truth=test_case.ground_truth_answer,
                    retrieved_chunks=pipeline_output.retrieved_chunks,
                )
            )

        # 4. Computed answer metrics
        tasks.extend([
            self.answer_evaluator.evaluate_citation_coverage(
                answer=pipeline_output.answer,
                citations=pipeline_output.citations,
            ),
            self.answer_evaluator.evaluate_hallucination(
                answer=pipeline_output.answer,
                retrieved_chunks=pipeline_output.retrieved_chunks,
            ),
            self.answer_evaluator.evaluate_answer_length(
                answer=pipeline_output.answer,
            ),
        ])

        # Token overlap (if ground truth available)
        if test_case.ground_truth_answer:
            tasks.append(
                self.answer_evaluator.evaluate_token_overlap(
                    prediction=pipeline_output.answer,
                    ground_truth=test_case.ground_truth_answer,
                )
            )

        # Execute all evaluations
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Add results to metrics
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed: {result}")
            else:
                metrics.add_result(result)

        logger.info(
            f"Evaluation complete for {test_case.id}: "
            f"score={metrics.overall_score:.3f}, passed={metrics.passed}"
        )

        return metrics

    async def evaluate_batch(
        self,
        test_cases: list[TestCase],
        pipeline_outputs: list[PipelineOutput],
    ) -> dict[str, Any]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test cases
            pipeline_outputs: Corresponding pipeline outputs

        Returns:
            Aggregated evaluation report
        """
        assert len(test_cases) == len(pipeline_outputs), "Mismatched test cases and outputs"

        logger.info(f"Starting batch evaluation of {len(test_cases)} test cases")

        # Evaluate all test cases
        tasks = [
            self.evaluate_pipeline(tc, po)
            for tc, po in zip(test_cases, pipeline_outputs)
        ]
        all_metrics = await asyncio.gather(*tasks)

        # Aggregate results
        report = self._generate_report(test_cases, all_metrics)

        logger.info(
            f"Batch evaluation complete: "
            f"{report['summary']['passed_count']}/{report['summary']['total_count']} passed "
            f"(avg score: {report['summary']['average_score']:.3f})"
        )

        return report

    async def _evaluate_retrieval(
        self,
        query: str,
        retrieved: list[str],
        ground_truth: list[str],
    ):
        """Helper to evaluate retrieval."""
        return await self.retrieval_evaluator.evaluate(
            query=query,
            prediction=retrieved,
            ground_truth=ground_truth,
        )

    def _generate_report(
        self,
        test_cases: list[TestCase],
        all_metrics: list[EvaluationMetrics],
    ) -> dict[str, Any]:
        """Generate comprehensive evaluation report."""
        total_count = len(all_metrics)
        passed_count = sum(1 for m in all_metrics if m.passed)
        failed_count = total_count - passed_count

        # Calculate average scores per metric type
        metric_type_scores = {}
        metric_type_counts = {}

        for metrics in all_metrics:
            for result in metrics.results:
                metric_type = result.metric_type.value
                if metric_type not in metric_type_scores:
                    metric_type_scores[metric_type] = 0.0
                    metric_type_counts[metric_type] = 0

                metric_type_scores[metric_type] += result.score
                metric_type_counts[metric_type] += 1

        # Average per metric type
        avg_scores_by_type = {
            metric_type: score / metric_type_counts[metric_type]
            for metric_type, score in metric_type_scores.items()
        }

        # Overall average
        average_score = sum(m.overall_score for m in all_metrics) / total_count if total_count > 0 else 0.0

        # Find failed test cases
        failed_tests = [
            {
                "test_case_id": m.test_case_id,
                "query": m.query,
                "score": m.overall_score,
                "failed_metrics": [
                    {
                        "name": r.metric_name,
                        "type": r.metric_type.value,
                        "score": r.score,
                        "threshold": r.threshold,
                        "explanation": r.explanation,
                    }
                    for r in m.get_failed_metrics()
                ],
            }
            for m in all_metrics if not m.passed
        ]

        return {
            "summary": {
                "total_count": total_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
                "average_score": average_score,
            },
            "scores_by_metric_type": avg_scores_by_type,
            "failed_tests": failed_tests,
            "all_results": [m.to_dict() for m in all_metrics],
        }
