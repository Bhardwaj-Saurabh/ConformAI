# ConformAI RAG Service

Advanced Agentic RAG service for EU AI compliance queries with query decomposition and ReAct agent.

## Features

- **Query Decomposition**: Automatically breaks complex queries into manageable sub-questions
- **ReAct Agent**: Iterative planning, acting, and observing for comprehensive answers
- **Legal-Grade Accuracy**: Citation enforcement and grounding validation
- **Hallucination Detection**: LLM-based verification to prevent unsupported claims
- **Multi-Tool Agent**: Retrieval, generation, synthesis, and validation tools
- **Safety Guardrails**: Scope checking and legal advice refusal

## Architecture

```
Query → Analysis → Decomposition → Safety Check →
ReAct Loop (max 5 iterations):
  ├─ Plan (decide next action)
  ├─ Act (execute tool)
  └─ Observe (check completion)
→ Synthesis → Grounding Validation → Response
```

## API Endpoints

### POST /api/v1/query

Process EU AI compliance query.

**Request**:
```json
{
  "query": "What are the documentation requirements for high-risk AI systems in recruitment?",
  "conversation_id": "optional-uuid",
  "max_iterations": 5
}
```

**Response**:
```json
{
  "success": true,
  "query": "...",
  "answer": "High-risk AI systems in recruitment must comply with...[Source 1: Article 11]...",
  "citations": [
    {
      "source_id": 1,
      "regulation": "EU AI Act",
      "article": "Article 11",
      "excerpt": "..."
    }
  ],
  "metadata": {
    "intent": "compliance_question",
    "ai_domain": "recruitment",
    "query_complexity": "complex",
    "processing_time_ms": 2500,
    "confidence_score": 0.92,
    "agent_iterations": 4
  },
  "reasoning_trace": [
    "Decomposed into 3 sub-questions",
    "Retrieved 12 legal chunks",
    "Answered all sub-questions",
    "Synthesized final answer"
  ],
  "agent_actions": [
    {
      "step": 1,
      "thought": "Need to retrieve sources about documentation requirements",
      "action": "retrieve_legal_chunks",
      "observation": "Retrieved 8 chunks"
    },
    ...
  ]
}
```

### GET /health

Health check.

## Running the Service

### Local Development

```bash
# Install dependencies
cd services/rag-service
uv pip install -e .

# Set environment variables
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

# Run service
python src/api/main.py
```

Service runs on `http://localhost:8001`

### Docker

```bash
docker build -t conformai-rag-service .
docker run -p 8001:8001 --env-file .env conformai-rag-service
```

## Agent Tools

The ReAct agent has access to these tools:

1. **retrieve_legal_chunks**: Search vector database for relevant legal documents
2. **answer_sub_question**: Generate grounded answer using retrieved sources
3. **synthesize_information**: Combine multiple sub-answers into coherent response
4. **validate_claim**: Verify if a claim is supported by sources
5. **search_specific_article**: Retrieve exact article by regulation and number

## Configuration

Key environment variables:

```bash
# LLM Configuration
LLM_PROVIDER=openai  # or anthropic
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# RAG Configuration
RETRIEVAL_TOP_K=10
RETRIEVAL_MIN_CONFIDENCE=0.6

# Agent Configuration
RAG_MAX_ITERATIONS=5  # Default in code
```

## Example Queries

**Simple Query**:
```
"What is a high-risk AI system?"
```
- No decomposition needed
- Single retrieval + generation
- ~1 second processing

**Complex Query**:
```
"Compare the documentation requirements for recruitment AI vs healthcare AI systems, and explain the key differences in their obligations."
```
- Decomposed into 4 sub-questions
- Multiple retrievals across iterations
- Synthesized comprehensive answer
- ~3-4 seconds processing

## Testing

Run unit tests:
```bash
pytest tests/unit/
```

Run integration tests:
```bash
pytest tests/integration/
```

## Architecture Details

### State Schema

The RAG state flows through all nodes and accumulates information:

- **Input**: query, conversation_id, user_context
- **Analysis**: intent, ai_domain, risk_category, complexity
- **Decomposition**: sub_queries with priority
- **Agent Loop**: actions, working_memory, iteration_count
- **Retrieval**: chunks, scores, history
- **Generation**: intermediate_answers, final_answer
- **Validation**: citations, grounding_validated
- **Safety**: is_safe, refusal_reason
- **Metadata**: processing_time, token_usage, confidence

### ReAct Agent Flow

1. **PLAN**: Analyze current state, decide next action
2. **ACT**: Execute selected tool with parameters
3. **OBSERVE**: Record results, check completion criteria
4. **Repeat** until done or max iterations reached

### Grounding Validation

Three-layer validation:

1. **Citation Completeness**: Check if answer has citations
2. **Citation Validity**: Verify citations reference valid chunks
3. **Hallucination Detection**: LLM-based verification of claims

Failed validation triggers regeneration (max 2 retries).

## Monitoring

The service includes:

- Structured JSON logging
- Processing time tracking
- Token usage counting
- Confidence scoring
- Agent action history
- Reasoning trace

## Error Handling

- Network errors: Retry with exponential backoff
- LLM errors: Fallback decisions
- Retrieval failures: Graceful degradation
- Validation failures: Regeneration or refusal

## Production Considerations

- Use Claude Sonnet for planning (better reasoning)
- Use GPT-4o-mini for generation (faster, cheaper)
- Set max_iterations based on latency requirements
- Monitor confidence scores and refusal rates
- Cache common queries if needed
- Rate limit API endpoints

## License

MIT
