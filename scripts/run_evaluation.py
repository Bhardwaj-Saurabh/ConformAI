#!/usr/bin/env python3
"""
Run RAG System Evaluation

This script runs evaluations on the ConformAI RAG system using Opik.
It creates datasets, runs evaluations, and stores results in Opik for tracking.

Usage:
    # Create all datasets in Opik
    python scripts/run_evaluation.py --create-datasets

    # Run evaluation on a specific dataset
    python scripts/run_evaluation.py --evaluate comprehensive-eval --experiment rag-v1.0

    # List available datasets
    python scripts/run_evaluation.py --list-datasets

    # Compare experiments
    python scripts/run_evaluation.py --compare rag-v1.0 rag-v1.1
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.evaluation.datasets import ALL_DATASETS, list_datasets
from shared.evaluation.metrics import get_default_scorers
from shared.evaluation.opik_evaluator import get_evaluator
from shared.utils import get_logger

logger = get_logger(__name__)


def create_all_datasets():
    """Create all evaluation datasets in Opik."""
    evaluator = get_evaluator()

    logger.info("Creating evaluation datasets in Opik...")

    for dataset_name, dataset_info in ALL_DATASETS.items():
        logger.info(f"\n📊 Creating dataset: {dataset_name}")
        logger.info(f"   Description: {dataset_info['description']}")
        logger.info(f"   Items: {len(dataset_info['items'])}")

        dataset_id = evaluator.create_dataset(
            dataset_name=dataset_name,
            items=dataset_info["items"],
            description=dataset_info["description"],
        )

        if dataset_id:
            logger.info(f"   ✓ Created with ID: {dataset_id}")
        else:
            logger.error(f"   ✗ Failed to create dataset")

    logger.info("\n✅ Dataset creation complete!")


def run_evaluation(dataset_name: str, experiment_name: str):
    """
    Run evaluation on a dataset.

    Args:
        dataset_name: Name of the dataset to evaluate
        experiment_name: Name for this experiment run
    """
    evaluator = get_evaluator()

    logger.info(f"\n🔬 Running evaluation: {experiment_name}")
    logger.info(f"   Dataset: {dataset_name}")

    # Define model function
    def rag_model(input_data: dict) -> dict:
        """Wrapper for RAG pipeline."""
        try:
            # Import here to avoid circular imports
            from services.rag_service.src.graph.graph import run_rag_pipeline

            query = input_data.get("input", "")
            result = asyncio.run(run_rag_pipeline(query))

            return {
                "output": result.get("final_answer", ""),
                "citations": result.get("citations", []),
                "confidence": result.get("confidence_score", 0.0),
            }
        except Exception as e:
            logger.error(f"Error in RAG model: {str(e)}")
            return {"output": "", "citations": [], "confidence": 0.0}

    # Get scoring functions
    scorers = get_default_scorers()

    # Run evaluation
    results = evaluator.evaluate(
        dataset_name=dataset_name,
        model_fn=rag_model,
        experiment_name=experiment_name,
        scoring_functions=scorers,
    )

    if results:
        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"   Experiment: {results['experiment_name']}")
        logger.info(f"   Timestamp: {results['timestamp']}")
    else:
        logger.error("\n❌ Evaluation failed!")


def compare_experiments(experiment_names: list[str]):
    """
    Compare multiple experiment results.

    Args:
        experiment_names: List of experiment names to compare
    """
    evaluator = get_evaluator()

    logger.info(f"\n📊 Comparing experiments:")
    for name in experiment_names:
        logger.info(f"   - {name}")

    results = evaluator.compare_experiments(experiment_names)

    if results:
        logger.info(f"\n✅ Comparison complete!")
        # Print comparison summary
        for exp_name in experiment_names:
            logger.info(f"\n{exp_name}:")
            # Would print metrics here
    else:
        logger.error("\n❌ Comparison failed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run RAG system evaluation using Opik"
    )

    parser.add_argument(
        "--create-datasets",
        action="store_true",
        help="Create all evaluation datasets in Opik",
    )

    parser.add_argument(
        "--list-datasets", action="store_true", help="List available datasets"
    )

    parser.add_argument(
        "--evaluate", type=str, metavar="DATASET", help="Run evaluation on dataset"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="evaluation",
        help="Experiment name for evaluation run",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="EXPERIMENT",
        help="Compare multiple experiments",
    )

    args = parser.parse_args()

    if args.create_datasets:
        create_all_datasets()

    elif args.list_datasets:
        datasets = list_datasets()
        logger.info("\n📚 Available datasets:")
        for name in datasets:
            info = ALL_DATASETS[name]
            logger.info(f"   - {name}: {info['description']} ({len(info['items'])} items)")

    elif args.evaluate:
        if args.evaluate not in ALL_DATASETS:
            logger.error(f"Dataset '{args.evaluate}' not found!")
            logger.info(f"Available datasets: {', '.join(list_datasets())}")
            sys.exit(1)

        run_evaluation(args.evaluate, args.experiment)

    elif args.compare:
        if len(args.compare) < 2:
            logger.error("Need at least 2 experiments to compare")
            sys.exit(1)

        compare_experiments(args.compare)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
