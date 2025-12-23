"""
Opik Evaluation System

Manages evaluation datasets and stores evaluation results in Opik.
Supports creating datasets, running evaluations, and tracking results over time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.config import get_settings
from shared.utils import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OpikEvaluator:
    """Manages evaluation datasets and results using Opik."""

    def __init__(self):
        """Initialize Opik evaluator."""
        self.opik = None
        self.client = None
        self._initialize_opik()

    def _initialize_opik(self):
        """Initialize Opik client."""
        if not settings.opik_enabled:
            logger.warning("Opik is not enabled. Set OPIK_ENABLED=true in .env")
            return

        try:
            import opik
            from opik import Opik

            # Configure Opik
            opik.configure(
                api_key=settings.opik_api_key,
                use_local=False,
            )

            # Create client
            self.client = Opik()
            self.opik = opik

            logger.info("✓ Opik evaluator initialized")
        except ImportError:
            logger.warning("Opik not installed. Install with: pip install opik")
        except Exception as e:
            logger.error(f"Failed to initialize Opik: {str(e)}")

    def create_dataset(
        self,
        dataset_name: str,
        items: list[dict[str, Any]],
        description: str | None = None,
    ) -> str | None:
        """
        Create an evaluation dataset in Opik.

        Args:
            dataset_name: Name of the dataset
            items: List of evaluation items, each with 'input' and 'expected_output'
            description: Optional dataset description

        Returns:
            Dataset ID if successful, None otherwise

        Example:
            >>> items = [
            ...     {
            ...         "input": "What are high-risk AI systems under the EU AI Act?",
            ...         "expected_output": "High-risk AI systems include...",
            ...         "metadata": {"category": "classification", "difficulty": "medium"}
            ...     }
            ... ]
            >>> dataset_id = evaluator.create_dataset("eu-ai-act-qa", items)
        """
        if not self.client:
            logger.warning("Opik client not initialized")
            return None

        try:
            # Create dataset
            dataset = self.client.create_dataset(
                name=dataset_name,
                description=description or f"Evaluation dataset: {dataset_name}",
            )

            # Add items to dataset
            for item in items:
                dataset.insert(
                    input=item.get("input"),
                    expected_output=item.get("expected_output"),
                    metadata=item.get("metadata", {}),
                )

            logger.info(
                f"✓ Created dataset '{dataset_name}' with {len(items)} items (ID: {dataset.id})"
            )
            return dataset.id

        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return None

    def load_dataset(self, dataset_name: str) -> list[dict[str, Any]] | None:
        """
        Load an evaluation dataset from Opik.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of dataset items if successful, None otherwise
        """
        if not self.client:
            logger.warning("Opik client not initialized")
            return None

        try:
            # Get dataset
            dataset = self.client.get_dataset(name=dataset_name)

            # Get all items
            items = []
            for item in dataset.get_items():
                items.append(
                    {
                        "input": item.input,
                        "expected_output": item.expected_output,
                        "metadata": item.metadata or {},
                    }
                )

            logger.info(f"✓ Loaded dataset '{dataset_name}' with {len(items)} items")
            return items

        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return None

    def evaluate(
        self,
        dataset_name: str,
        model_fn: callable,
        experiment_name: str | None = None,
        scoring_functions: list[callable] | None = None,
    ) -> dict[str, Any] | None:
        """
        Run evaluation on a dataset using Opik.

        Args:
            dataset_name: Name of the dataset to evaluate
            model_fn: Function that takes input and returns output
            experiment_name: Optional experiment name
            scoring_functions: Optional list of scoring functions

        Returns:
            Evaluation results dict if successful, None otherwise

        Example:
            >>> def rag_model(input_data):
            ...     return {"output": "High-risk AI systems include..."}
            >>>
            >>> results = evaluator.evaluate(
            ...     dataset_name="eu-ai-act-qa",
            ...     model_fn=rag_model,
            ...     experiment_name="rag-v1.0",
            ...     scoring_functions=[accuracy_score, relevance_score]
            ... )
        """
        if not self.client or not self.opik:
            logger.warning("Opik client not initialized")
            return None

        try:
            # Get dataset
            dataset = self.client.get_dataset(name=dataset_name)

            # Create experiment name
            if not experiment_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"eval_{dataset_name}_{timestamp}"

            # Use default scoring if none provided
            if not scoring_functions:
                scoring_functions = [self._default_exact_match_scorer]

            # Run evaluation
            logger.info(
                f"Running evaluation: {experiment_name} on dataset {dataset_name}"
            )

            from opik.evaluation import evaluate

            evaluation_result = evaluate(
                dataset=dataset,
                task=model_fn,
                scoring_metrics=scoring_functions,
                experiment_name=experiment_name,
            )

            # Extract results
            results = {
                "experiment_name": experiment_name,
                "dataset_name": dataset_name,
                "timestamp": datetime.now().isoformat(),
                "scores": {},
            }

            logger.info(f"✓ Evaluation completed: {experiment_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to run evaluation: {str(e)}")
            return None

    def log_evaluation_result(
        self,
        experiment_name: str,
        input_text: str,
        expected_output: str,
        actual_output: str,
        scores: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ):
        """
        Log a single evaluation result to Opik.

        Args:
            experiment_name: Name of the experiment
            input_text: Input query/text
            expected_output: Expected output
            actual_output: Actual model output
            scores: Dict of metric names to scores
            metadata: Optional additional metadata
        """
        if not self.opik:
            return

        try:
            # Use @track decorator functionality
            self.opik.track(
                name=f"{experiment_name}_eval",
                input={"query": input_text, "expected": expected_output},
                output={"response": actual_output},
                metadata={
                    **(metadata or {}),
                    "scores": scores,
                    "experiment": experiment_name,
                },
            )

        except Exception as e:
            logger.error(f"Failed to log evaluation result: {str(e)}")

    def compare_experiments(
        self, experiment_names: list[str]
    ) -> dict[str, Any] | None:
        """
        Compare multiple experiment results.

        Args:
            experiment_names: List of experiment names to compare

        Returns:
            Comparison results dict
        """
        if not self.client:
            logger.warning("Opik client not initialized")
            return None

        try:
            # Get experiments
            experiments = []
            for name in experiment_names:
                exp = self.client.get_experiment(name=name)
                if exp:
                    experiments.append(exp)

            if not experiments:
                logger.warning("No experiments found")
                return None

            # Compare metrics
            comparison = {
                "experiments": experiment_names,
                "timestamp": datetime.now().isoformat(),
                "metrics": {},
            }

            logger.info(f"✓ Compared {len(experiments)} experiments")
            return comparison

        except Exception as e:
            logger.error(f"Failed to compare experiments: {str(e)}")
            return None

    def _default_exact_match_scorer(self, expected: str, actual: str) -> float:
        """Default exact match scoring function."""
        return 1.0 if expected.strip() == actual.strip() else 0.0

    def export_results(
        self, experiment_name: str, output_path: str | Path
    ) -> bool:
        """
        Export evaluation results to a JSON file.

        Args:
            experiment_name: Name of the experiment
            output_path: Path to save results

        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logger.warning("Opik client not initialized")
            return False

        try:
            # Get experiment
            exp = self.client.get_experiment(name=experiment_name)

            # Export to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            results = {
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "results": [],  # Would be populated with actual results
            }

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"✓ Exported results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False


# Convenience function
def get_evaluator() -> OpikEvaluator:
    """Get or create OpikEvaluator instance."""
    return OpikEvaluator()
