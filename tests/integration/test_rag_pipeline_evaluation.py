"""
Integration tests for RAG pipeline with comprehensive evaluation.

Run with: pytest tests/integration/test_rag_pipeline_evaluation.py -v
"""

import json
import pytest
import asyncio
from pathlib import Path
from typing import List

from tests.evaluation.pipeline_evaluator import PipelineEvaluator, TestCase, PipelineOutput
from graph.graph import run_rag_pipeline
from shared.utils.logger import get_logger

logger = get_logger(__name__)


class TestRAGPipelineEvaluation:
    """Integration tests with evaluation metrics."""

    @pytest.fixture
    def test_cases(self) -> List[TestCase]:
        """Load test cases from golden dataset."""
        dataset_path = Path(__file__).parent.parent / "test_datasets" / "golden_qa_eu_ai_act.json"

        with open(dataset_path, "r") as f:
            data = json.load(f)

        return [
            TestCase(
                id=item["id"],
                query=item["query"],
                ground_truth_answer=item.get("ground_truth_answer"),
                relevant_chunk_ids=item.get("relevant_chunk_ids"),
                expected_aspects=item.get("expected_aspects"),
                metadata={"difficulty": item.get("difficulty"), "category": item.get("category")},
            )
            for item in data
        ]

    @pytest.fixture
    def evaluator(self) -> PipelineEvaluator:
        """Create pipeline evaluator."""
        return PipelineEvaluator(
            retrieval_threshold=0.7,
            answer_threshold=0.7,
            judge_model="claude-3-5-sonnet-20241022",
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_single_query_evaluation(self, test_cases, evaluator):
        """Test evaluation of a single query."""
        # Use first easy test case
        test_case = next(tc for tc in test_cases if tc.metadata.get("difficulty") == "easy")

        logger.info(f"Testing query: {test_case.query}")

        # Run RAG pipeline
        result = await run_rag_pipeline(test_case.query)

        # Create pipeline output
        pipeline_output = PipelineOutput(
            query=test_case.query,
            answer=result.get("final_answer", ""),
            retrieved_chunk_ids=[],  # Would need to extract from result
            retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
            citations=result.get("citations", []),
            metadata={
                "processing_time_ms": result.get("processing_time_ms", 0),
                "confidence_score": result.get("confidence_score", 0.0),
                "iterations": result.get("iteration_count", 0),
            },
        )

        # Evaluate
        metrics = await evaluator.evaluate_pipeline(test_case, pipeline_output)

        # Assertions
        assert metrics.overall_score > 0.0, "Should have non-zero overall score"
        assert len(metrics.results) > 0, "Should have evaluation results"

        # Log results
        logger.info(f"Overall score: {metrics.overall_score:.3f}")
        logger.info(f"Passed: {metrics.passed}")

        for result in metrics.results:
            logger.info(
                f"  {result.metric_name}: {result.score:.3f} "
                f"({'PASS' if result.passed else 'FAIL'})"
            )

        # Check specific metrics
        faithfulness_results = [r for r in metrics.results if r.metric_name == "faithfulness"]
        assert len(faithfulness_results) > 0, "Should have faithfulness evaluation"

        relevance_results = [r for r in metrics.results if r.metric_name == "relevance"]
        assert len(relevance_results) > 0, "Should have relevance evaluation"

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_batch_evaluation(self, test_cases, evaluator):
        """Test batch evaluation of multiple queries."""
        # Use first 3 test cases for faster testing
        test_subset = test_cases[:3]

        logger.info(f"Testing {len(test_subset)} queries in batch")

        # Run RAG pipeline for all test cases
        pipeline_outputs = []
        for test_case in test_subset:
            result = await run_rag_pipeline(test_case.query)

            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer=result.get("final_answer", ""),
                retrieved_chunk_ids=[],
                retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
                citations=result.get("citations", []),
                metadata={
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "confidence_score": result.get("confidence_score", 0.0),
                },
            )
            pipeline_outputs.append(pipeline_output)

        # Batch evaluation
        report = await evaluator.evaluate_batch(test_subset, pipeline_outputs)

        # Assertions
        assert report["summary"]["total_count"] == len(test_subset)
        assert report["summary"]["average_score"] > 0.0
        assert "scores_by_metric_type" in report

        # Log report
        logger.info(f"Batch evaluation results:")
        logger.info(f"  Total: {report['summary']['total_count']}")
        logger.info(f"  Passed: {report['summary']['passed_count']}")
        logger.info(f"  Failed: {report['summary']['failed_count']}")
        logger.info(f"  Pass rate: {report['summary']['pass_rate']:.1%}")
        logger.info(f"  Average score: {report['summary']['average_score']:.3f}")

        logger.info(f"\nScores by metric type:")
        for metric_type, score in report["scores_by_metric_type"].items():
            logger.info(f"  {metric_type}: {score:.3f}")

        # Check pass rate threshold (should pass at least 60% of tests)
        assert report["summary"]["pass_rate"] >= 0.6, "Pass rate should be at least 60%"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retrieval_quality(self, test_cases, evaluator):
        """Test retrieval quality specifically."""
        test_case = test_cases[0]  # First test case

        result = await run_rag_pipeline(test_case.query)

        # Check retrieval metrics
        assert len(result.get("all_retrieved_chunks", [])) > 0, "Should retrieve chunks"
        assert result.get("all_retrieved_chunks", [])[0].get("score", 0) > 0, "Chunks should have scores"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_citation_quality(self, test_cases, evaluator):
        """Test citation quality."""
        test_case = test_cases[1]  # Documentation requirements test

        result = await run_rag_pipeline(test_case.query)

        # Check citations
        citations = result.get("citations", [])
        assert len(citations) > 0, "Should have citations"

        # Create pipeline output for citation evaluation
        pipeline_output = PipelineOutput(
            query=test_case.query,
            answer=result.get("final_answer", ""),
            retrieved_chunk_ids=[],
            retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
            citations=citations,
            metadata={},
        )

        # Evaluate citation quality
        citation_result = await evaluator.llm_judge.evaluate_citation_quality(
            answer=pipeline_output.answer,
            citations=pipeline_output.citations,
            retrieved_chunks=pipeline_output.retrieved_chunks,
        )

        logger.info(f"Citation quality score: {citation_result.score:.3f}")
        logger.info(f"Explanation: {citation_result.explanation}")

        assert citation_result.score > 0.0, "Should have non-zero citation quality score"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_faithfulness_check(self, test_cases, evaluator):
        """Test faithfulness evaluation."""
        test_case = test_cases[2]  # Recruitment AI test

        result = await run_rag_pipeline(test_case.query)

        # Evaluate faithfulness
        faithfulness_result = await evaluator.llm_judge.evaluate_faithfulness(
            query=test_case.query,
            answer=result.get("final_answer", ""),
            retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
        )

        logger.info(f"Faithfulness score: {faithfulness_result.score:.3f}")
        logger.info(f"Explanation: {faithfulness_result.explanation}")

        # Should be faithful (grounded in sources)
        assert faithfulness_result.score >= 0.6, "Answer should be reasonably faithful to sources"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_evaluation_report_generation(self, test_cases, evaluator):
        """Test that evaluation generates comprehensive report."""
        test_subset = test_cases[:2]

        # Run pipeline
        pipeline_outputs = []
        for test_case in test_subset:
            result = await run_rag_pipeline(test_case.query)
            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer=result.get("final_answer", ""),
                retrieved_chunk_ids=[],
                retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
                citations=result.get("citations", []),
                metadata={},
            )
            pipeline_outputs.append(pipeline_output)

        # Generate report
        report = await evaluator.evaluate_batch(test_subset, pipeline_outputs)

        # Verify report structure
        assert "summary" in report
        assert "scores_by_metric_type" in report
        assert "failed_tests" in report
        assert "all_results" in report

        # Save report to file for review
        report_path = Path(__file__).parent.parent / "evaluation_reports" / "test_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to: {report_path}")


# CI/CD thresholds
class TestCICDGates:
    """Tests that enforce CI/CD quality gates."""

    @pytest.fixture
    def evaluator(self) -> PipelineEvaluator:
        """Create evaluator with strict thresholds for CI/CD."""
        return PipelineEvaluator(
            retrieval_threshold=0.75,  # Higher threshold for CI/CD
            answer_threshold=0.75,
        )

    @pytest.mark.asyncio
    @pytest.mark.cicd
    async def test_minimum_faithfulness_threshold(self, evaluator):
        """Ensure answers meet minimum faithfulness threshold."""
        query = "What are the prohibited AI practices under the EU AI Act?"

        result = await run_rag_pipeline(query)

        faithfulness_result = await evaluator.llm_judge.evaluate_faithfulness(
            query=query,
            answer=result.get("final_answer", ""),
            retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
        )

        # CI/CD gate: faithfulness must be >= 0.75
        assert faithfulness_result.score >= 0.75, (
            f"Faithfulness score {faithfulness_result.score:.3f} below CI/CD threshold 0.75. "
            f"Reason: {faithfulness_result.explanation}"
        )

    @pytest.mark.asyncio
    @pytest.mark.cicd
    async def test_minimum_relevance_threshold(self, evaluator):
        """Ensure answers meet minimum relevance threshold."""
        query = "What are the prohibited AI practices under the EU AI Act?"

        result = await run_rag_pipeline(query)

        relevance_result = await evaluator.llm_judge.evaluate_relevance(
            query=query,
            answer=result.get("final_answer", ""),
        )

        # CI/CD gate: relevance must be >= 0.75
        assert relevance_result.score >= 0.75, (
            f"Relevance score {relevance_result.score:.3f} below CI/CD threshold 0.75. "
            f"Reason: {relevance_result.explanation}"
        )

    @pytest.mark.asyncio
    @pytest.mark.cicd
    async def test_minimum_citation_coverage(self, evaluator):
        """Ensure answers have sufficient citations."""
        query = "What documentation requirements apply to high-risk AI systems?"

        result = await run_rag_pipeline(query)

        citation_result = await evaluator.answer_evaluator.evaluate_citation_coverage(
            answer=result.get("final_answer", ""),
            citations=result.get("citations", []),
        )

        # CI/CD gate: citation coverage must be >= 0.5 (at least 0.5 citations per sentence)
        assert citation_result.score >= 0.5, (
            f"Citation coverage {citation_result.score:.3f} below CI/CD threshold 0.5"
        )
