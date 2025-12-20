#!/usr/bin/env python3
"""
Command-line script to run RAG pipeline evaluation.

Usage:
    python tests/evaluation/run_evaluation.py --dataset golden --num-cases 5
    python tests/evaluation/run_evaluation.py --dataset golden --cicd --strict
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.evaluation.pipeline_evaluator import PipelineEvaluator, TestCase, PipelineOutput
from tests.evaluation.report_generator import EvaluationReportGenerator
from services.rag_service.src.graph.graph import run_rag_pipeline
from shared.utils.logger import get_logger

logger = get_logger(__name__)


async def load_test_cases(dataset_name: str, num_cases: int = None) -> List[TestCase]:
    """
    Load test cases from dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'golden')
        num_cases: Number of test cases to load (None for all)

    Returns:
        List of test cases
    """
    dataset_path = Path(__file__).parent.parent / "test_datasets" / f"golden_qa_{dataset_name}.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Limit number of cases if specified
    if num_cases is not None:
        data = data[:num_cases]

    test_cases = [
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

    logger.info(f"Loaded {len(test_cases)} test cases from {dataset_name}")
    return test_cases


async def run_pipeline_for_test_cases(test_cases: List[TestCase]) -> List[PipelineOutput]:
    """
    Run RAG pipeline for all test cases.

    Args:
        test_cases: List of test cases

    Returns:
        List of pipeline outputs
    """
    pipeline_outputs = []

    for i, test_case in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}/{len(test_cases)}: {test_case.id}")

        try:
            # Run RAG pipeline
            result = await run_rag_pipeline(test_case.query)

            # Create pipeline output
            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer=result.get("final_answer", ""),
                retrieved_chunk_ids=[],
                retrieved_chunks=[
                    chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])
                ],
                citations=result.get("citations", []),
                metadata={
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "iterations": result.get("iteration_count", 0),
                },
            )
            pipeline_outputs.append(pipeline_output)

        except Exception as e:
            logger.error(f"Error processing test case {test_case.id}: {e}", exc_info=True)

            # Create empty output with error
            pipeline_output = PipelineOutput(
                query=test_case.query,
                answer="",
                retrieved_chunk_ids=[],
                retrieved_chunks=[],
                citations=[],
                metadata={"error": str(e)},
            )
            pipeline_outputs.append(pipeline_output)

    return pipeline_outputs


async def run_evaluation(
    dataset_name: str = "eu_ai_act",
    num_cases: int = None,
    retrieval_threshold: float = 0.7,
    answer_threshold: float = 0.7,
    cicd_mode: bool = False,
    strict_mode: bool = False,
    output_dir: Path = None,
) -> dict:
    """
    Run complete evaluation.

    Args:
        dataset_name: Name of dataset to use
        num_cases: Number of test cases (None for all)
        retrieval_threshold: Threshold for retrieval metrics
        answer_threshold: Threshold for answer metrics
        cicd_mode: Enable CI/CD mode with strict thresholds
        strict_mode: Use even stricter thresholds
        output_dir: Output directory for reports

    Returns:
        Evaluation report dictionary
    """
    logger.info("=" * 80)
    logger.info("Starting RAG Pipeline Evaluation")
    logger.info("=" * 80)

    # Adjust thresholds for CI/CD and strict mode
    if strict_mode:
        retrieval_threshold = 0.8
        answer_threshold = 0.8
        logger.info("Strict mode enabled: thresholds set to 0.8")
    elif cicd_mode:
        retrieval_threshold = 0.75
        answer_threshold = 0.75
        logger.info("CI/CD mode enabled: thresholds set to 0.75")

    # Load test cases
    logger.info(f"Loading test cases from dataset: {dataset_name}")
    test_cases = await load_test_cases(dataset_name, num_cases)

    # Run pipeline
    logger.info("Running RAG pipeline for test cases...")
    pipeline_outputs = await run_pipeline_for_test_cases(test_cases)

    # Create evaluator
    evaluator = PipelineEvaluator(
        retrieval_threshold=retrieval_threshold,
        answer_threshold=answer_threshold,
    )

    # Run evaluation
    logger.info("Evaluating results...")
    report = await evaluator.evaluate_batch(test_cases, pipeline_outputs)

    # Generate detailed report
    logger.info("Generating detailed report...")
    report_generator = EvaluationReportGenerator(output_dir=output_dir)

    # Get all metrics
    all_metrics = await asyncio.gather(
        *[evaluator.evaluate_pipeline(tc, po) for tc, po in zip(test_cases, pipeline_outputs)]
    )

    # Generate and save report
    from datetime import datetime

    report_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    detailed_report = report_generator.generate_report(
        all_metrics=all_metrics,
        report_name=report_name,
        include_charts=True,
    )

    # Export CSV
    csv_path = report_generator.export_to_csv(all_metrics)
    logger.info(f"CSV export saved to: {csv_path}")

    # Generate HTML report
    html_path = report_generator.generate_html_report(detailed_report)
    logger.info(f"HTML report saved to: {html_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("Evaluation Summary")
    logger.info("=" * 80)
    logger.info(f"Total test cases: {report['summary']['total_count']}")
    logger.info(f"Passed: {report['summary']['passed_count']}")
    logger.info(f"Failed: {report['summary']['failed_count']}")
    logger.info(f"Pass rate: {report['summary']['pass_rate']:.1%}")
    logger.info(f"Average score: {report['summary']['average_score']:.3f}")
    logger.info("=" * 80)

    # Print scores by metric type
    logger.info("Scores by Metric Type:")
    for metric_type, score in report.get("scores_by_metric_type", {}).items():
        logger.info(f"  {metric_type}: {score:.3f}")
    logger.info("=" * 80)

    # CI/CD mode: enforce pass rate threshold
    if cicd_mode:
        required_pass_rate = 0.6  # 60% minimum
        if report["summary"]["pass_rate"] < required_pass_rate:
            logger.error(
                f"CI/CD FAILURE: Pass rate {report['summary']['pass_rate']:.1%} "
                f"below required threshold {required_pass_rate:.1%}"
            )
            sys.exit(1)

        # Enforce minimum average score
        required_avg_score = 0.7
        if report["summary"]["average_score"] < required_avg_score:
            logger.error(
                f"CI/CD FAILURE: Average score {report['summary']['average_score']:.3f} "
                f"below required threshold {required_avg_score:.3f}"
            )
            sys.exit(1)

        logger.info("✅ CI/CD quality gates PASSED")

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")

    parser.add_argument(
        "--dataset",
        type=str,
        default="eu_ai_act",
        help="Dataset name to use (default: eu_ai_act)",
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=None,
        help="Number of test cases to evaluate (default: all)",
    )

    parser.add_argument(
        "--retrieval-threshold",
        type=float,
        default=0.7,
        help="Retrieval metrics threshold (default: 0.7)",
    )

    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=0.7,
        help="Answer metrics threshold (default: 0.7)",
    )

    parser.add_argument(
        "--cicd",
        action="store_true",
        help="Enable CI/CD mode with stricter thresholds (0.75)",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode with very strict thresholds (0.8)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports (default: tests/evaluation_reports)",
    )

    args = parser.parse_args()

    # Run evaluation
    try:
        report = asyncio.run(
            run_evaluation(
                dataset_name=args.dataset,
                num_cases=args.num_cases,
                retrieval_threshold=args.retrieval_threshold,
                answer_threshold=args.answer_threshold,
                cicd_mode=args.cicd,
                strict_mode=args.strict,
                output_dir=args.output_dir,
            )
        )

        logger.info("Evaluation completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
