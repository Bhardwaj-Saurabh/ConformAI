# ConformAI Evaluation Framework

Comprehensive evaluation system for the ConformAI RAG pipeline with LLM-as-a-Judge, retrieval metrics, and CI/CD integration.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Metrics](#metrics)
- [Usage](#usage)
- [CI/CD Integration](#cicd-integration)
- [Dashboard](#dashboard)
- [Test Datasets](#test-datasets)
- [Reports](#reports)

---

## Overview

The ConformAI evaluation framework provides comprehensive assessment of the RAG pipeline across multiple dimensions:

- **Retrieval Quality**: Precision@K, Recall@K, MRR, NDCG
- **Answer Faithfulness**: LLM-judged grounding in sources
- **Answer Relevance**: LLM-judged relevance to query
- **Answer Correctness**: LLM-judged factual accuracy
- **Answer Completeness**: Coverage of expected aspects
- **Citation Quality**: Quality and distribution of citations
- **Performance Metrics**: Processing time, confidence scores

The system uses **LLM-as-a-Judge** (Claude 3.5 Sonnet) for subjective evaluations and computed metrics for objective measurements.

---

## Architecture

```
tests/evaluation/
├── base.py                     # Base classes and enums
├── llm_judge.py                # LLM-as-a-Judge evaluator
├── retrieval_metrics.py        # Retrieval quality metrics
├── answer_metrics.py           # Answer quality metrics
├── pipeline_evaluator.py       # End-to-end pipeline evaluator
├── report_generator.py         # Report generation with visualizations
├── run_evaluation.py           # CLI script for running evaluations
└── README.md                   # This file

tests/test_datasets/
└── golden_qa_eu_ai_act.json   # Golden test dataset (10 cases)

tests/integration/
└── test_rag_pipeline_evaluation.py  # Pytest integration tests

tests/evaluation_reports/       # Generated reports (JSON, CSV, HTML)
```

---

## Metrics

### 1. Retrieval Metrics (Computed)

**Source**: `retrieval_metrics.py` - `RetrievalEvaluator`

| Metric | Description | Range | Threshold |
|--------|-------------|-------|-----------|
| **Precision@K** | Proportion of retrieved chunks that are relevant | 0.0-1.0 | 0.7 |
| **Recall@K** | Proportion of relevant chunks that were retrieved | 0.0-1.0 | 0.7 |
| **MRR** | Mean Reciprocal Rank - rank of first relevant chunk | 0.0-1.0 | 0.7 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | 0.0-1.0 | 0.7 |
| **Hit Rate** | Whether any relevant chunk was retrieved | 0.0-1.0 | 1.0 |

**Example**:
```python
from tests.evaluation.retrieval_metrics import RetrievalEvaluator

evaluator = RetrievalEvaluator(k=10, threshold=0.7)
result = await evaluator.evaluate(
    query="What are prohibited AI practices?",
    prediction=["chunk_1", "chunk_5", "chunk_3"],  # Retrieved IDs
    ground_truth=["chunk_1", "chunk_3", "chunk_7"],  # Relevant IDs
)

print(f"Precision@10: {result.details['precision_at_k']}")
print(f"Recall@10: {result.details['recall_at_k']}")
print(f"MRR: {result.details['mrr']}")
```

---

### 2. LLM-as-a-Judge Metrics

**Source**: `llm_judge.py` - `LLMJudge`

Uses Claude 3.5 Sonnet as an expert judge for subjective quality assessment.

#### Faithfulness

Evaluates whether the answer is grounded in retrieved sources without hallucinations.

**Criteria**:
- Every claim must be supported by sources
- No hallucinated information
- Proper attribution
- No speculation beyond sources

**Prompt Template**:
```
You are evaluating the FAITHFULNESS of an AI system's answer.

QUERY: {query}
ANSWER: {answer}
SOURCES: {retrieved_chunks}

Rate faithfulness from 0.0 to 1.0 based on:
1. Every claim is supported by sources
2. No hallucinations
3. Proper attribution

Return JSON: {"score": 0.0-1.0, "reasoning": "...", "strengths": [...], "weaknesses": [...]}
```

**Example**:
```python
from tests.evaluation.llm_judge import LLMJudge

judge = LLMJudge(threshold=0.7)
result = await judge.evaluate_faithfulness(
    query="What are prohibited AI practices?",
    answer="The EU AI Act prohibits AI systems that deploy subliminal techniques...",
    retrieved_chunks=["Article 5 states that AI systems deploying subliminal techniques..."],
)

print(f"Faithfulness: {result.score}")
print(f"Reasoning: {result.explanation}")
```

#### Relevance

Evaluates how well the answer addresses the query.

**Criteria**:
- Direct response to the question
- Focus on requested information
- No irrelevant content

#### Correctness

Evaluates factual accuracy against ground truth.

**Criteria**:
- Factually correct claims
- No contradictions with ground truth
- Accurate representation of legal provisions

#### Completeness

Evaluates coverage of expected aspects.

**Criteria**:
- Covers all expected aspects
- Comprehensive response
- No missing critical information

#### Citation Quality

Evaluates quality and distribution of citations.

**Criteria**:
- Citations support claims
- Proper citation format
- Appropriate distribution throughout answer

---

### 3. Answer Quality Metrics (Computed)

**Source**: `answer_metrics.py` - `AnswerEvaluator`

#### Citation Coverage

Measures the proportion of sentences with citations.

**Formula**: `citations / sentences`

**Good threshold**: ≥ 0.5 (at least 0.5 citations per sentence)

#### Token Overlap (F1)

Measures token-level similarity with ground truth answer.

**Formula**: `F1 = 2 * (precision * recall) / (precision + recall)`

#### Source Grounding Score

Measures proportion of answer tokens found in sources (hallucination detection).

**Formula**: `tokens_in_sources / total_answer_tokens`

**High score** = Low hallucination risk

#### Answer Length Appropriateness

Checks if answer length is appropriate.

**Default range**: 50-500 words

---

## Usage

### 1. Command Line (Quick Evaluation)

Run evaluation from the command line:

```bash
# Run on full golden dataset
python tests/evaluation/run_evaluation.py --dataset eu_ai_act

# Run on 5 test cases
python tests/evaluation/run_evaluation.py --dataset eu_ai_act --num-cases 5

# CI/CD mode with stricter thresholds (0.75)
python tests/evaluation/run_evaluation.py --dataset eu_ai_act --cicd

# Strict mode with very strict thresholds (0.8)
python tests/evaluation/run_evaluation.py --dataset eu_ai_act --strict

# Custom thresholds
python tests/evaluation/run_evaluation.py \
    --dataset eu_ai_act \
    --retrieval-threshold 0.75 \
    --answer-threshold 0.8 \
    --num-cases 10
```

**Output**:
- JSON report in `tests/evaluation_reports/`
- CSV export for data analysis
- HTML report for viewing
- Console summary

---

### 2. Python API (Programmatic)

Use the evaluation framework programmatically:

```python
import asyncio
from tests.evaluation.pipeline_evaluator import PipelineEvaluator, TestCase, PipelineOutput
from services.rag_service.src.graph.graph import run_rag_pipeline

# Create test case
test_case = TestCase(
    id="test_001",
    query="What are the prohibited AI practices under the EU AI Act?",
    ground_truth_answer="The EU AI Act prohibits...",
    relevant_chunk_ids=["eu_ai_act_article_5"],
    expected_aspects=["subliminal techniques", "social scoring"],
)

# Run RAG pipeline
result = await run_rag_pipeline(test_case.query)

# Create pipeline output
pipeline_output = PipelineOutput(
    query=test_case.query,
    answer=result.get("final_answer", ""),
    retrieved_chunk_ids=[],
    retrieved_chunks=[chunk.get("content", "") for chunk in result.get("all_retrieved_chunks", [])],
    citations=result.get("citations", []),
    metadata={},
)

# Evaluate
evaluator = PipelineEvaluator(retrieval_threshold=0.7, answer_threshold=0.7)
metrics = await evaluator.evaluate_pipeline(test_case, pipeline_output)

print(f"Overall score: {metrics.overall_score:.3f}")
print(f"Passed: {metrics.passed}")

for result in metrics.results:
    print(f"{result.metric_name}: {result.score:.3f} ({'PASS' if result.passed else 'FAIL'})")
```

---

### 3. Pytest (Integration Tests)

Run evaluation as part of pytest test suite:

```bash
# Run all evaluation tests
pytest tests/integration/test_rag_pipeline_evaluation.py -v

# Run only CI/CD quality gate tests
pytest tests/integration/test_rag_pipeline_evaluation.py -m cicd -v

# Run with coverage
pytest tests/integration/test_rag_pipeline_evaluation.py --cov -v

# Run slow tests (batch evaluation)
pytest tests/integration/test_rag_pipeline_evaluation.py -m slow -v
```

**Available test markers**:
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.cicd` - CI/CD quality gates
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.asyncio` - Async tests

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: RAG Evaluation

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .

      - name: Run CI/CD evaluation
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python tests/evaluation/run_evaluation.py --cicd --num-cases 5

      - name: Upload evaluation reports
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-reports
          path: tests/evaluation_reports/
```

### Quality Gates

**CI/CD mode enforces**:
- Minimum pass rate: **60%**
- Minimum average score: **0.70**
- Retrieval threshold: **0.75**
- Answer threshold: **0.75**

**Strict mode enforces**:
- Minimum pass rate: **70%**
- Minimum average score: **0.75**
- Retrieval threshold: **0.80**
- Answer threshold: **0.80**

---

## Dashboard

### Streamlit Dashboard

Interactive dashboard for viewing and analyzing evaluation results.

**Launch**:
```bash
streamlit run app.py
```

**Features**:
1. **Run New Evaluation**
   - Configure thresholds
   - Select test dataset
   - Choose number of test cases
   - View real-time progress

2. **View Saved Reports**
   - Browse historical reports
   - Compare performance over time
   - Export to JSON/CSV

3. **Visualizations**
   - Score distribution
   - Metrics by type (bar charts)
   - Pass/fail breakdown
   - Failed test analysis

**Navigation**:
- Main app: Query testing interface
- Page 1: Evaluation Dashboard (📊 Evaluation Dashboard)

---

## Test Datasets

### Golden EU AI Act Q&A Dataset

**Location**: `tests/test_datasets/golden_qa_eu_ai_act.json`

**Coverage**: 10 carefully curated test cases covering:
- EU AI Act provisions
- GDPR compliance
- Risk classifications
- Prohibited practices
- Documentation requirements
- Transparency obligations

**Difficulty Levels**:
- **Easy**: Simple factual queries (3 cases)
- **Medium**: Multi-aspect queries (4 cases)
- **Hard**: Complex compliance scenarios (3 cases)

**Categories**:
- `prohibitions` - Prohibited AI practices
- `obligations` - Provider obligations
- `risk_classification` - Risk category determination
- `transparency` - Transparency requirements
- `definitions` - Legal definitions
- `compliance_intersection` - GDPR + AI Act

**Example Test Case**:
```json
{
  "id": "eu_ai_act_001",
  "query": "What are the prohibited AI practices under the EU AI Act?",
  "ground_truth_answer": "The EU AI Act prohibits several AI practices including...",
  "relevant_chunk_ids": ["eu_ai_act_article_5", "eu_ai_act_article_5_1a"],
  "expected_aspects": ["subliminal techniques", "exploitation of vulnerabilities"],
  "difficulty": "medium",
  "category": "prohibitions"
}
```

### Creating Custom Datasets

Create your own test dataset:

```python
import json

dataset = [
    {
        "id": "custom_001",
        "query": "Your query here",
        "ground_truth_answer": "Expected answer",
        "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
        "expected_aspects": ["aspect1", "aspect2"],
        "difficulty": "medium",
        "category": "custom_category",
    }
]

with open("tests/test_datasets/custom_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

---

## Reports

### Report Structure

Generated reports include:

```json
{
  "report_name": "evaluation_20250120_143022",
  "generated_at": "2025-01-20T14:30:22Z",
  "summary": {
    "total_count": 10,
    "passed_count": 8,
    "failed_count": 2,
    "pass_rate": 0.8,
    "average_score": 0.78,
    "median_score": 0.80,
    "min_score": 0.45,
    "max_score": 0.95
  },
  "metric_analysis": {
    "retrieval": {
      "average_score": 0.82,
      "pass_rate": 0.9,
      "total_evaluations": 10
    },
    "faithfulness": {
      "average_score": 0.85,
      "pass_rate": 0.9,
      "total_evaluations": 10
    }
  },
  "failed_tests": [
    {
      "test_case_id": "eu_ai_act_007",
      "query": "...",
      "overall_score": 0.65,
      "failed_metrics": [...]
    }
  ],
  "chart_data": {...},
  "all_results": [...]
}
```

### Report Formats

1. **JSON** - Machine-readable, full detail
2. **CSV** - Tabular format for analysis
3. **HTML** - Human-readable with formatting

### Report Location

Default: `tests/evaluation_reports/`

**Files generated**:
- `evaluation_YYYYMMDD_HHMMSS.json`
- `evaluation_export_YYYYMMDD_HHMMSS.csv`
- `evaluation_YYYYMMDD_HHMMSS.html`

---

## Best Practices

### 1. Test Dataset Management

- Keep golden datasets version-controlled
- Include diverse difficulty levels
- Cover all critical use cases
- Update when regulations change

### 2. Threshold Configuration

**Development**: Use lenient thresholds (0.6-0.7)
```python
evaluator = PipelineEvaluator(retrieval_threshold=0.6, answer_threshold=0.6)
```

**CI/CD**: Use moderate thresholds (0.75)
```python
evaluator = PipelineEvaluator(retrieval_threshold=0.75, answer_threshold=0.75)
```

**Production**: Use strict thresholds (0.8+)
```python
evaluator = PipelineEvaluator(retrieval_threshold=0.8, answer_threshold=0.8)
```

### 3. Continuous Monitoring

- Run evaluations on every PR
- Track metrics over time
- Set up alerts for regressions
- Review failed tests regularly

### 4. LLM Judge Usage

**Pros**:
- Captures subjective quality
- Nuanced evaluation
- Detailed explanations

**Cons**:
- Slower than computed metrics
- Costs API calls
- May have variance

**Recommendation**: Combine LLM judge with computed metrics for comprehensive evaluation.

### 5. Interpreting Results

**High Faithfulness + Low Relevance**: Answer is grounded but doesn't address the query
**High Relevance + Low Faithfulness**: Answer addresses query but hallucinates
**Low Retrieval + High Answer**: Answer is good despite poor retrieval (memorization risk)
**High Retrieval + Low Answer**: Good retrieval but poor synthesis

---

## Troubleshooting

### Issue: LLM Judge Fails

**Solution**:
- Check API key is set: `ANTHROPIC_API_KEY`
- Verify network connectivity
- Check rate limits
- Review LLM judge prompts

### Issue: Low Retrieval Scores

**Solution**:
- Verify ground truth chunk IDs are correct
- Check vector store has indexed documents
- Review embedding quality
- Adjust retrieval parameters

### Issue: Evaluation Timeout

**Solution**:
- Reduce number of test cases
- Increase timeout in async calls
- Run evaluations in smaller batches

---

## Contributing

To add new metrics:

1. Create evaluator class inheriting from `BaseEvaluator`
2. Implement `evaluate()` method returning `EvaluationResult`
3. Add to `PipelineEvaluator` evaluation tasks
4. Update test dataset if needed
5. Document metric in this README

---

## License

MIT License - See main project LICENSE file

---

## Contact

For questions or issues with the evaluation framework, please open an issue on GitHub or contact the ConformAI team.
