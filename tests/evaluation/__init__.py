"""
Evaluation framework for ConformAI RAG system.

This module provides comprehensive evaluation metrics for:
- Retrieval accuracy (precision, recall, MRR, NDCG)
- Answer quality (faithfulness, relevance, correctness, completeness)
- Tool use effectiveness
- End-to-end pipeline performance
- LLM-as-a-Judge evaluation
"""

from .answer_metrics import AnswerEvaluator
from .base import BaseEvaluator, EvaluationMetrics, EvaluationResult
from .llm_judge import LLMJudge
from .pipeline_evaluator import PipelineEvaluator
from .retrieval_metrics import RetrievalEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "RetrievalEvaluator",
    "AnswerEvaluator",
    "LLMJudge",
    "PipelineEvaluator",
]
