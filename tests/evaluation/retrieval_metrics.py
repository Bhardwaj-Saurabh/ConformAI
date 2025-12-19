"""Retrieval evaluation metrics (precision, recall, MRR, NDCG)."""

import math
from typing import Any, Dict, List, Optional, Set

from tests.evaluation.base import BaseEvaluator, EvaluationResult, MetricType


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for retrieval quality.

    Metrics:
    - Precision@K: How many retrieved chunks are relevant
    - Recall@K: How many relevant chunks were retrieved
    - Mean Reciprocal Rank (MRR): Rank of first relevant chunk
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Hit Rate: Whether any relevant chunk was retrieved
    """

    def __init__(self, k: int = 10, threshold: float = 0.7):
        """
        Initialize retrieval evaluator.

        Args:
            k: Top-k chunks to evaluate
            threshold: Minimum score to pass
        """
        super().__init__(threshold)
        self.k = k

    async def evaluate(
        self,
        query: str,
        prediction: List[str],  # Retrieved chunk IDs
        ground_truth: List[str],  # Relevant chunk IDs
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate retrieval quality."""
        # Calculate all metrics
        precision = self.calculate_precision_at_k(prediction, ground_truth, self.k)
        recall = self.calculate_recall_at_k(prediction, ground_truth, self.k)
        mrr = self.calculate_mrr(prediction, ground_truth)
        ndcg = self.calculate_ndcg_at_k(prediction, ground_truth, self.k)
        hit_rate = self.calculate_hit_rate(prediction, ground_truth)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Overall score (average of key metrics)
        overall_score = (precision + recall + mrr + ndcg) / 4.0

        return self._create_result(
            metric_name="retrieval_quality",
            metric_type=MetricType.RETRIEVAL,
            score=overall_score,
            details={
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_score": f1,
                "mrr": mrr,
                "ndcg_at_k": ndcg,
                "hit_rate": hit_rate,
                "k": self.k,
                "num_retrieved": len(prediction),
                "num_relevant": len(ground_truth),
                "num_relevant_retrieved": len(set(prediction[:self.k]) & set(ground_truth)),
            },
            explanation=f"Precision@{self.k}: {precision:.3f}, Recall@{self.k}: {recall:.3f}, MRR: {mrr:.3f}, NDCG@{self.k}: {ndcg:.3f}",
        )

    @staticmethod
    def calculate_precision_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# relevant items in top-k) / k
        """
        if not retrieved or k == 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_retrieved = sum(1 for item in top_k if item in relevant_set)
        return relevant_retrieved / min(k, len(top_k))

    @staticmethod
    def calculate_recall_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# relevant items in top-k) / (total # relevant items)
        """
        if not relevant:
            return 0.0

        if not retrieved:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        relevant_retrieved = sum(1 for item in top_k if item in relevant_set)
        return relevant_retrieved / len(relevant)

    @staticmethod
    def calculate_mrr(
        retrieved: List[str],
        relevant: List[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / (rank of first relevant item)
        """
        if not retrieved or not relevant:
            return 0.0

        relevant_set = set(relevant)

        for rank, item in enumerate(retrieved, 1):
            if item in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def calculate_ndcg_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.

        NDCG@K = DCG@K / IDCG@K

        Args:
            retrieved: Retrieved items (ordered by rank)
            relevant: Relevant items
            k: Top-k items
            relevance_scores: Optional relevance scores (0-1), defaults to binary
        """
        if not retrieved or not relevant or k == 0:
            return 0.0

        top_k = retrieved[:k]
        relevant_set = set(relevant)

        # Binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}

        # Calculate DCG@K
        dcg = 0.0
        for rank, item in enumerate(top_k, 1):
            if item in relevant_set:
                relevance = relevance_scores.get(item, 0.0)
                dcg += relevance / math.log2(rank + 1)

        # Calculate IDCG@K (ideal DCG)
        ideal_ranking = sorted(relevant, key=lambda x: relevance_scores.get(x, 0.0), reverse=True)
        idcg = 0.0
        for rank, item in enumerate(ideal_ranking[:k], 1):
            relevance = relevance_scores.get(item, 0.0)
            idcg += relevance / math.log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def calculate_hit_rate(
        retrieved: List[str],
        relevant: List[str],
    ) -> float:
        """
        Calculate Hit Rate (binary).

        Hit Rate = 1 if any relevant item retrieved, 0 otherwise
        """
        if not retrieved or not relevant:
            return 0.0

        relevant_set = set(relevant)
        return 1.0 if any(item in relevant_set for item in retrieved) else 0.0

    async def evaluate_batch(
        self,
        queries: List[str],
        predictions: List[List[str]],
        ground_truths: List[List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality across multiple queries.

        Returns:
            Dictionary of average metrics
        """
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        hit_rates = []

        for query, pred, gt in zip(queries, predictions, ground_truths):
            result = await self.evaluate(query, pred, gt)
            details = result.details

            precisions.append(details["precision_at_k"])
            recalls.append(details["recall_at_k"])
            mrrs.append(details["mrr"])
            ndcgs.append(details["ndcg_at_k"])
            hit_rates.append(details["hit_rate"])

        return {
            f"avg_precision_at_{self.k}": sum(precisions) / len(precisions) if precisions else 0.0,
            f"avg_recall_at_{self.k}": sum(recalls) / len(recalls) if recalls else 0.0,
            "avg_mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            f"avg_ndcg_at_{self.k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
            "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0.0,
        }
