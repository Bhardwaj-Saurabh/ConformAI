"""
Evaluation framework for ConformAI RAG system.

This module provides comprehensive evaluation metrics for:
- Retrieval accuracy (precision, recall, MRR, NDCG)
- Answer quality (faithfulness, relevance, correctness, completeness)
- Tool use effectiveness
- End-to-end pipeline performance
- LLM-as-a-Judge evaluation
"""

from .base import BaseEvaluator, EvaluationResult, EvaluationMetrics
from .retrieval_metrics import RetrievalEvaluator
from .answer_metrics import AnswerEvaluator
from .llm_judge import LLMJudge
from .pipeline_evaluator import PipelineEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "RetrievalEvaluator",
    "AnswerEvaluator",
    "LLMJudge",
    "PipelineEvaluator",
]
