"""LLM-as-a-Judge evaluator for subjective metrics."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.evaluation.base import BaseEvaluator, EvaluationResult, MetricType
from services.rag_service.src.llm.client import get_llm_client, invoke_llm


class LLMJudgement(BaseModel):
    """Structured output from LLM judge."""

    score: float  # 0.0 to 1.0
    reasoning: str
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]


class LLMJudge(BaseEvaluator):
    """
    LLM-as-a-Judge evaluator.

    Uses a powerful LLM to evaluate RAG outputs on subjective metrics like:
    - Faithfulness (grounding in sources)
    - Relevance (answers the question)
    - Correctness (factual accuracy)
    - Completeness (covers all aspects)
    - Citation quality
    """

    def __init__(self, threshold: float = 0.7, judge_model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize LLM judge.

        Args:
            threshold: Minimum score to pass
            judge_model: Model to use as judge
        """
        super().__init__(threshold)
        self.judge_model = judge_model
        self.llm = get_llm_client(provider="anthropic", model=judge_model, temperature=0.0)

    async def evaluate_faithfulness(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[str],
    ) -> EvaluationResult:
        """
        Evaluate if the answer is faithful to retrieved sources.

        Checks:
        - All claims are supported by sources
        - No hallucinations
        - Proper attribution
        """
        prompt = f"""You are evaluating the FAITHFULNESS of an AI system's answer to a legal compliance query.

**Query**: {query}

**Answer**: {answer}

**Retrieved Sources**:
{self._format_sources(retrieved_chunks)}

**Task**: Evaluate if the answer is fully grounded in the provided sources.

**Criteria**:
1. Every claim in the answer must be supported by the sources
2. No information should be added that isn't in the sources
3. No hallucinations or unsupported interpretations
4. Citations should be accurate

**Output Format** (JSON):
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Provide ONLY the JSON output, no additional text."""

        judgement = await self._get_llm_judgement(prompt)

        return self._create_result(
            metric_name="faithfulness",
            metric_type=MetricType.FAITHFULNESS,
            score=judgement.score,
            details={
                "strengths": judgement.strengths,
                "weaknesses": judgement.weaknesses,
                "suggestions": judgement.improvement_suggestions,
            },
            explanation=judgement.reasoning,
        )

    async def evaluate_relevance(
        self,
        query: str,
        answer: str,
    ) -> EvaluationResult:
        """
        Evaluate if the answer is relevant to the query.

        Checks:
        - Directly addresses the question
        - Stays on topic
        - Provides useful information
        """
        prompt = f"""You are evaluating the RELEVANCE of an AI system's answer to a legal compliance query.

**Query**: {query}

**Answer**: {answer}

**Task**: Evaluate if the answer is relevant and directly addresses the query.

**Criteria**:
1. Answer directly addresses the question asked
2. Stays on topic throughout
3. Provides information that is useful for the query
4. Doesn't include irrelevant tangents

**Output Format** (JSON):
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Provide ONLY the JSON output, no additional text."""

        judgement = await self._get_llm_judgement(prompt)

        return self._create_result(
            metric_name="relevance",
            metric_type=MetricType.RELEVANCE,
            score=judgement.score,
            details={
                "strengths": judgement.strengths,
                "weaknesses": judgement.weaknesses,
                "suggestions": judgement.improvement_suggestions,
            },
            explanation=judgement.reasoning,
        )

    async def evaluate_correctness(
        self,
        query: str,
        answer: str,
        ground_truth: Optional[str] = None,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate factual correctness of the answer.

        Checks:
        - Facts are accurate
        - No misinformation
        - Aligns with ground truth (if provided)
        """
        sources_section = ""
        if retrieved_chunks:
            sources_section = f"""
**Retrieved Sources**:
{self._format_sources(retrieved_chunks)}
"""

        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"""
**Ground Truth Answer**: {ground_truth}
"""

        prompt = f"""You are evaluating the CORRECTNESS of an AI system's answer to a legal compliance query.

**Query**: {query}

**Answer**: {answer}
{ground_truth_section}
{sources_section}

**Task**: Evaluate if the answer is factually correct.

**Criteria**:
1. Facts are accurate and verifiable
2. No misinformation or incorrect statements
3. Legal interpretations are sound
4. {"Aligns with ground truth answer" if ground_truth else "Aligns with source documents"}

**Output Format** (JSON):
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Provide ONLY the JSON output, no additional text."""

        judgement = await self._get_llm_judgement(prompt)

        return self._create_result(
            metric_name="correctness",
            metric_type=MetricType.CORRECTNESS,
            score=judgement.score,
            details={
                "strengths": judgement.strengths,
                "weaknesses": judgement.weaknesses,
                "suggestions": judgement.improvement_suggestions,
            },
            explanation=judgement.reasoning,
        )

    async def evaluate_completeness(
        self,
        query: str,
        answer: str,
        expected_aspects: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate if the answer is complete.

        Checks:
        - Covers all aspects of the question
        - Provides sufficient detail
        - Doesn't miss important information
        """
        aspects_section = ""
        if expected_aspects:
            aspects_section = f"""
**Expected Aspects to Cover**:
{chr(10).join(f"- {aspect}" for aspect in expected_aspects)}
"""

        prompt = f"""You are evaluating the COMPLETENESS of an AI system's answer to a legal compliance query.

**Query**: {query}

**Answer**: {answer}
{aspects_section}

**Task**: Evaluate if the answer is complete and covers all aspects of the question.

**Criteria**:
1. Addresses all parts of the multi-part query
2. Provides sufficient detail and explanation
3. Doesn't miss important related information
4. {"Covers all expected aspects" if expected_aspects else "Comprehensively answers the question"}

**Output Format** (JSON):
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Provide ONLY the JSON output, no additional text."""

        judgement = await self._get_llm_judgement(prompt)

        return self._create_result(
            metric_name="completeness",
            metric_type=MetricType.COMPLETENESS,
            score=judgement.score,
            details={
                "strengths": judgement.strengths,
                "weaknesses": judgement.weaknesses,
                "suggestions": judgement.improvement_suggestions,
            },
            explanation=judgement.reasoning,
        )

    async def evaluate_citation_quality(
        self,
        answer: str,
        citations: List[Dict[str, Any]],
        retrieved_chunks: List[str],
    ) -> EvaluationResult:
        """
        Evaluate quality of citations.

        Checks:
        - Citations are present
        - Citations are accurate
        - Citations support claims
        """
        prompt = f"""You are evaluating the CITATION QUALITY of an AI system's answer to a legal compliance query.

**Answer**: {answer}

**Citations Provided**:
{json.dumps(citations, indent=2)}

**Available Sources**:
{self._format_sources(retrieved_chunks)}

**Task**: Evaluate the quality and accuracy of citations.

**Criteria**:
1. All major claims have citations
2. Citations accurately reference the sources
3. Citations support the claims they're attached to
4. Citation format is clear and useful

**Output Format** (JSON):
{{
    "score": <float 0.0-1.0>,
    "reasoning": "<detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}

Provide ONLY the JSON output, no additional text."""

        judgement = await self._get_llm_judgement(prompt)

        return self._create_result(
            metric_name="citation_quality",
            metric_type=MetricType.CITATION_QUALITY,
            score=judgement.score,
            details={
                "strengths": judgement.strengths,
                "weaknesses": judgement.weaknesses,
                "suggestions": judgement.improvement_suggestions,
                "citation_count": len(citations),
            },
            explanation=judgement.reasoning,
        )

    async def evaluate(
        self,
        query: str,
        prediction: Any,
        ground_truth: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Run all LLM-based evaluations."""
        # This is a wrapper - use specific evaluate_* methods instead
        raise NotImplementedError("Use specific evaluate_* methods (evaluate_faithfulness, evaluate_relevance, etc.)")

    async def _get_llm_judgement(self, prompt: str) -> LLMJudgement:
        """Get structured judgement from LLM."""
        response = await invoke_llm(self.llm, [HumanMessage(content=prompt)])

        # Parse JSON response
        try:
            judgement_dict = json.loads(response.content)
            return LLMJudgement(**judgement_dict)
        except (json.JSONDecodeError, Exception) as e:
            # Fallback to neutral score
            return LLMJudgement(
                score=0.5,
                reasoning=f"Failed to parse LLM response: {str(e)}",
                strengths=[],
                weaknesses=["Failed to get structured evaluation"],
                improvement_suggestions=[],
            )

    def _format_sources(self, chunks: List[str]) -> str:
        """Format retrieved chunks for display."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[Source {i}]: {chunk[:500]}...")  # First 500 chars
        return "\n\n".join(formatted)
