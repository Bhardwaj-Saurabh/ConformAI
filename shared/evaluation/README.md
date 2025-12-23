# ConformAI Evaluation System

Comprehensive evaluation framework for the ConformAI RAG system using Opik for dataset management and result tracking.

## Overview

This evaluation system provides:
- **Pre-defined evaluation datasets** covering EU AI regulation topics
- **Custom scoring metrics** tailored for compliance Q&A
- **Opik integration** for experiment tracking and comparison
- **Automated evaluation pipeline** with result storage

## Quick Start

### 1. Setup Opik

Ensure Opik is configured in your `.env`:
```bash
OPIK_ENABLED=true
OPIK_API_KEY=your_api_key
COMET_WORKSPACE=your_workspace
OPIK_PROJECT_NAME=conformai
```

### 2. Create Evaluation Datasets

```bash
# Create all pre-defined datasets in Opik
python scripts/run_evaluation.py --create-datasets
```

This creates 6 datasets:
- `eu-ai-act-classification` - EU AI Act risk classification
- `gdpr-ai-data` - GDPR and AI training data
- `compliance-obligations` - Compliance requirements
- `biometric-systems` - Biometric AI regulations
- `generative-ai` - Generative AI obligations
- `comprehensive-eval` - Combined dataset (all above)

### 3. Run Evaluation

```bash
# Run evaluation on comprehensive dataset
python scripts/run_evaluation.py --evaluate comprehensive-eval --experiment rag-v1.0

# Run on specific dataset
python scripts/run_evaluation.py --evaluate eu-ai-act-classification --experiment rag-v1.0
```

### 4. View Results in Opik

1. Go to https://www.comet.com/
2. Navigate to your workspace
3. Select the ConformAI project
4. View experiments and metrics

## Architecture

```
shared/evaluation/
├── __init__.py              # Module exports
├── opik_evaluator.py        # Opik integration and evaluation runner
├── datasets.py              # Pre-defined evaluation datasets
├── metrics.py               # Scoring functions
└── README.md                # This file

scripts/
└── run_evaluation.py        # CLI tool for running evaluations
```

## Evaluation Datasets

### Dataset Format

Each evaluation item contains:
```python
{
    "input": "Question or query",
    "expected_output": "Expected answer",
    "metadata": {
        "category": "classification|prohibition|obligations|...",
        "regulation": "EU AI Act|GDPR",
        "article": "Article reference",
        "difficulty": "easy|medium|hard"
    }
}
```

### Available Datasets

| Dataset Name | Description | Items | Topics |
|--------------|-------------|-------|--------|
| `eu-ai-act-classification` | Risk classification questions | 3 | High-risk systems, prohibitions |
| `gdpr-ai-data` | GDPR compliance for AI | 2 | Training data, individual rights |
| `compliance-obligations` | Documentation and oversight | 2 | Requirements, human oversight |
| `biometric-systems` | Biometric AI regulations | 2 | Facial recognition, emotion AI |
| `generative-ai` | Generative AI obligations | 1 | Model requirements |
| `comprehensive-eval` | All topics combined | 10 | Complete coverage |

### Adding Custom Datasets

Edit `shared/evaluation/datasets.py`:

```python
CUSTOM_DATASET = [
    {
        "input": "Your question here",
        "expected_output": "Expected answer",
        "metadata": {
            "category": "custom",
            "regulation": "Regulation name",
            "difficulty": "medium"
        }
    }
]

# Add to ALL_DATASETS
ALL_DATASETS["custom-dataset"] = {
    "description": "Description here",
    "items": CUSTOM_DATASET
}
```

Then create in Opik:
```bash
python scripts/run_evaluation.py --create-datasets
```

## Scoring Metrics

### Built-in Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `exact_match_score` | Exact string match | 0.0 - 1.0 |
| `contains_keywords_score` | Keyword overlap | 0.0 - 1.0 |
| `regulation_mention_score` | Correct regulation citations | 0.0 - 1.0 |
| `citation_presence_score` | Citations provided | 0.0 or 1.0 |
| `answer_length_score` | Length similarity | 0.0 - 1.0 |
| `semantic_similarity_score` | Semantic overlap | 0.0 - 1.0 |
| `comprehensive_score` | Weighted combination | 0.0 - 1.0 |

### Default Scoring Configuration

```python
weights = {
    "keywords": 0.3,      # Keyword presence
    "regulation": 0.25,   # Regulation mentions
    "semantic": 0.25,     # Semantic similarity
    "length": 0.1,        # Answer length
    "citations": 0.1      # Citation presence
}
```

### Custom Scoring Functions

Create Opik-compatible scorers:

```python
class CustomScorer:
    def __init__(self):
        self.name = "custom_score"

    def __call__(self, output: dict) -> float:
        expected = output.get("expected_output", "")
        actual = output.get("output", "")

        # Your scoring logic
        score = compute_score(expected, actual)

        return score
```

Use in evaluation:
```python
from shared.evaluation import get_evaluator

evaluator = get_evaluator()
evaluator.evaluate(
    dataset_name="my-dataset",
    model_fn=my_model,
    scoring_functions=[CustomScorer()]
)
```

## Programmatic Usage

### Create Dataset

```python
from shared.evaluation import get_evaluator

evaluator = get_evaluator()

items = [
    {
        "input": "What are prohibited AI practices?",
        "expected_output": "Prohibited practices include...",
        "metadata": {"category": "prohibition"}
    }
]

dataset_id = evaluator.create_dataset(
    dataset_name="my-eval-dataset",
    items=items,
    description="Custom evaluation dataset"
)
```

### Load Dataset

```python
items = evaluator.load_dataset("my-eval-dataset")

for item in items:
    print(f"Q: {item['input']}")
    print(f"A: {item['expected_output']}")
```

### Run Evaluation

```python
def rag_model(input_data):
    query = input_data.get("input")
    # Run your RAG pipeline
    result = run_rag_pipeline(query)
    return {
        "output": result["final_answer"],
        "citations": result["citations"]
    }

from shared.evaluation.metrics import get_default_scorers

results = evaluator.evaluate(
    dataset_name="comprehensive-eval",
    model_fn=rag_model,
    experiment_name="rag-v2.0",
    scoring_functions=get_default_scorers()
)
```

### Log Individual Result

```python
evaluator.log_evaluation_result(
    experiment_name="rag-v1.0",
    input_text="What are high-risk AI systems?",
    expected_output="High-risk systems include...",
    actual_output="According to the EU AI Act, high-risk...",
    scores={
        "keywords": 0.85,
        "regulation": 0.90,
        "citations": 1.0
    },
    metadata={"model": "gpt-4o-mini", "temperature": 0.0}
)
```

### Compare Experiments

```python
comparison = evaluator.compare_experiments([
    "rag-v1.0",
    "rag-v1.1",
    "rag-v2.0"
])

print(f"Best performing: {comparison['best_experiment']}")
print(f"Metrics: {comparison['metrics']}")
```

### Export Results

```python
evaluator.export_results(
    experiment_name="rag-v1.0",
    output_path="results/rag-v1.0-results.json"
)
```

## CLI Usage

### List Available Datasets

```bash
python scripts/run_evaluation.py --list-datasets
```

Output:
```
📚 Available datasets:
   - eu-ai-act-classification: Questions about EU AI Act risk classification (3 items)
   - gdpr-ai-data: Questions about GDPR and AI training data (2 items)
   ...
```

### Create Datasets in Opik

```bash
python scripts/run_evaluation.py --create-datasets
```

### Run Evaluation

```bash
# Evaluate specific dataset
python scripts/run_evaluation.py \
    --evaluate comprehensive-eval \
    --experiment rag-v1.0

# Evaluate with custom experiment name
python scripts/run_evaluation.py \
    --evaluate eu-ai-act-classification \
    --experiment "rag-gpt4-temp0"
```

### Compare Experiments

```bash
python scripts/run_evaluation.py --compare rag-v1.0 rag-v1.1 rag-v2.0
```

## Integration with RAG Pipeline

### Automatic Evaluation After Deployment

Add to CI/CD pipeline (`.github/workflows/deploy.yml`):

```yaml
- name: Run Evaluation
  run: |
    python scripts/run_evaluation.py \
      --evaluate comprehensive-eval \
      --experiment "deployment-${{ github.sha }}"
```

### Continuous Evaluation

Set up scheduled evaluation:

```yaml
# .github/workflows/scheduled-eval.yml
name: Scheduled Evaluation

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run evaluation
        run: |
          python scripts/run_evaluation.py \
            --evaluate comprehensive-eval \
            --experiment "weekly-$(date +%Y%m%d)"
```

## Best Practices

### 1. Version Your Experiments

Use semantic versioning:
```bash
python scripts/run_evaluation.py --evaluate comprehensive-eval --experiment rag-v1.0.0
python scripts/run_evaluation.py --evaluate comprehensive-eval --experiment rag-v1.1.0
```

### 2. Tag Experiments with Metadata

Include relevant metadata:
- Model version
- Temperature
- Prompt template version
- Retrieval parameters

### 3. Regular Evaluation

- Run evaluation before each deployment
- Schedule weekly comprehensive evaluations
- Compare against baseline

### 4. Track Regressions

Monitor metrics over time:
```python
baseline_score = 0.85
current_score = results["weighted_average"]

if current_score < baseline_score - 0.05:
    logger.warning("⚠️ Performance regression detected!")
```

### 5. Expand Datasets

Continuously add edge cases and failure modes:
- User-reported issues
- Difficult questions
- Ambiguous scenarios
- Multi-hop reasoning

## Metrics Dashboard in Opik

Opik provides:
1. **Experiment Comparison** - Side-by-side metric comparison
2. **Time Series** - Track metrics over time
3. **Trace Visualization** - Inspect individual predictions
4. **Error Analysis** - Identify failure patterns
5. **Dataset Versioning** - Track dataset changes

Access at: https://www.comet.com/[workspace]/projects/conformai

## Troubleshooting

### Opik Not Initialized

```
WARNING: Opik client not initialized
```

**Solution:**
1. Check `.env` file has correct credentials
2. Verify `OPIK_ENABLED=true`
3. Test connection: `python -c "import opik; opik.configure(api_key='...')"`

### Dataset Creation Failed

```
ERROR: Failed to create dataset
```

**Solution:**
1. Verify Opik API key is valid
2. Check workspace name is correct
3. Ensure dataset name is unique

### Evaluation Timeout

```
ERROR: Evaluation timed out
```

**Solution:**
1. Reduce dataset size for testing
2. Increase timeout in evaluation code
3. Check RAG service is running

## Next Steps

1. **Create Datasets**: `python scripts/run_evaluation.py --create-datasets`
2. **Run First Evaluation**: `python scripts/run_evaluation.py --evaluate comprehensive-eval --experiment baseline`
3. **View in Opik**: Check results at https://www.comet.com/
4. **Iterate**: Improve RAG system and re-evaluate
5. **Compare**: Track improvements over time

## Reference

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Evaluation Best Practices](https://www.comet.com/docs/opik/evaluation/)
- [Metrics Guide](./metrics.py)
- [Dataset Examples](./datasets.py)
