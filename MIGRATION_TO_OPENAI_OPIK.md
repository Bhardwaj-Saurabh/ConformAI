# Migration to OpenAI Embeddings & Opik Observability

## Summary of Changes

This document outlines the migration from Sentence Transformers to OpenAI embeddings, and from LangSmith to Opik for observability.

---

## 1. Embedding Model Changes

### Before (Sentence Transformers)
- **Model**: BAAI/bge-large-en-v1.5
- **Dimension**: 1024
- **Hosting**: Local (CPU/GPU)
- **Dependencies**: `sentence-transformers`, `transformers`, `torch`

### After (OpenAI)
- **Model**: text-embedding-3-large
- **Dimension**: 1024 (adjustable 256-3072)
- **Hosting**: OpenAI API
- **Dependencies**: `openai`

### Benefits
✅ No local GPU/CPU required for embeddings
✅ Consistent quality across environments
✅ Reduced Docker image size (no PyTorch)
✅ Scalable via API
✅ Latest embedding technology from OpenAI

### Cost Considerations
- **text-embedding-3-large**: $0.13 per 1M tokens
- **text-embedding-3-small**: $0.02 per 1M tokens (1536 dim)

For 10,000 chunks averaging 200 tokens each:
- Cost: ~$0.26 for large model
- Cost: ~$0.04 for small model

---

## 2. Updated Files

### Core Embedding Generator
**File**: `services/data-pipeline/src/embeddings/embedding_generator.py`

**Changes**:
- Replaced `SentenceTransformer` with `OpenAI` client
- Updated `generate_embeddings()` to use OpenAI API with batching
- Added rate limiting (0.1s between batches)
- Removed local model save/load methods (not applicable for API)
- Updated example usage

**Usage**:
```python
from services.data_pipeline.src.embeddings import EmbeddingGenerator

# Initialize with OpenAI
generator = EmbeddingGenerator(
    model_name="text-embedding-3-large",
    batch_size=100,
    dimensions=1024,  # Optional: reduce from 3072 default
)

# Generate embeddings
chunks_with_embeddings = generator.generate_embeddings(chunks)
```

### Configuration
**File**: `shared/config/settings.py`

**Changes**:
```python
# Old
embedding_model: str = "BAAI/bge-large-en-v1.5"
embedding_dimension: int = 1024
embedding_device: str = "cpu"
langsmith_api_key: str = ""
langchain_tracing_v2: bool = False

# New
embedding_model: str = "text-embedding-3-large"
embedding_dimension: int = 1024
embedding_provider: Literal["openai"] = "openai"
opik_api_key: str = ""
opik_enabled: bool = False
```

### Environment Variables
**File**: `.env.example`

**Changes**:
```bash
# Old
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=false
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu

# New
OPIK_API_KEY=your_opik_api_key_here
OPIK_ENABLED=false
OPIK_WORKSPACE=conformai
OPIK_PROJECT=eu-compliance-rag
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_PROVIDER=openai
```

### Dependencies
**File**: `pyproject.toml`

**Removed**:
- `langsmith>=0.1.0`
- `sentence-transformers>=3.0.0` (optional - still needed for chunker tokenization)
- `transformers>=4.45.0` (optional)
- `torch>=2.1.0` (optional)

**Added**:
- `opik>=1.0.0`

---

## 3. Opik Observability Integration

### New Utility Module
**File**: `shared/utils/opik_tracer.py`

Provides decorators and utilities for tracking:
- LLM calls
- Embedding generation
- Custom operations
- Metrics and events

**Decorators**:
```python
from shared.utils import track_operation, track_llm_call, track_embedding_call

@track_operation("parse_document", "parsing")
def parse_document(file_path):
    ...

@track_llm_call("claude-3-5-sonnet-20241022", "anthropic")
def generate_answer(prompt):
    ...

@track_embedding_call("text-embedding-3-large")
def generate_embeddings(texts):
    ...
```

**Manual Logging**:
```python
from shared.utils import log_metric, log_event

# Log metrics
log_metric("chunks_processed", 1250, {"regulation": "GDPR"})

# Log events
log_event("document_indexed", {
    "celex": "32016R0679",
    "chunks": 145,
    "status": "success"
})
```

---

## 4. Migration Steps

### For Development

1. **Update Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add:
   OPENAI_API_KEY=your_openai_key_here
   OPIK_API_KEY=your_opik_key_here  # Optional
   OPIK_ENABLED=true  # Optional
   ```

2. **Install Dependencies**:
   ```bash
   uv sync
   ```

3. **Update Qdrant Collection** (if dimension changed):
   ```bash
   # Delete old collection
   python -c "from services.data_pipeline.src.indexing import QdrantIndexer; QdrantIndexer().client.delete_collection('eu_legal_documents_development')"

   # Will auto-create with new dimensions on first index
   ```

4. **Test Embedding Generation**:
   ```bash
   python services/data-pipeline/src/embeddings/embedding_generator.py
   ```

### For Production

1. **Update Environment Variables** in deployment (Kubernetes secrets, etc.)

2. **Re-index All Documents**:
   - Old embeddings (1024 dim from BGE) won't match OpenAI embeddings
   - Must regenerate all embeddings
   - Use Airflow DAG: `document_processing`

3. **Monitor Costs**:
   - Set up billing alerts in OpenAI dashboard
   - Monitor usage via Opik dashboard

---

## 5. Opik Setup

### Get API Key

1. Sign up at [https://www.comet.com/opik](https://www.comet.com/opik)
2. Create a new workspace: `conformai`
3. Get API key from settings
4. Add to `.env`:
   ```bash
   OPIK_API_KEY=your_key_here
   OPIK_ENABLED=true
   ```

### View Traces

1. Visit Opik dashboard
2. Select workspace: `conformai`
3. Select project: `eu-compliance-rag`
4. View:
   - LLM calls with prompts and responses
   - Embedding generation stats
   - Operation traces with timing
   - Custom metrics and events

---

## 6. Backward Compatibility

If you need to temporarily use Sentence Transformers:

1. Install old dependencies:
   ```bash
   uv add sentence-transformers transformers torch
   ```

2. Create alternate embedding generator class (legacy)

3. Update configuration to switch providers

**Note**: Not recommended - better to fully migrate.

---

## 7. Cost Optimization Tips

### Reduce Embedding Dimensions
```python
# Use smaller dimensions for cost savings
generator = EmbeddingGenerator(
    model_name="text-embedding-3-large",
    dimensions=512,  # Instead of 3072
)
```

### Use Smaller Model
```python
# Switch to text-embedding-3-small (87% cheaper)
generator = EmbeddingGenerator(
    model_name="text-embedding-3-small",  # 1536 dim, $0.02/1M tokens
)
```

### Batch Efficiently
```python
# Already implemented - batches of 100 texts per API call
generator = EmbeddingGenerator(batch_size=100)
```

### Cache Embeddings
- Embeddings are deterministic for same text
- Store in Qdrant permanently
- Only regenerate when text changes

---

## 8. Testing

### Unit Tests
```bash
# Test embedding generation
python -m pytest tests/unit/test_embedding_generator.py

# Test Opik integration
python -m pytest tests/unit/test_opik_tracer.py
```

### Integration Tests
```bash
# Test full pipeline
python scripts/test_data_pipeline.py
```

### Manual Test
```bash
# Generate sample embeddings
python services/data-pipeline/src/embeddings/embedding_generator.py
```

---

## 9. Troubleshooting

### OpenAI API Errors

**Rate Limits**:
- Solution: Increase delay between batches in `generate_embeddings()`
- Or: Use smaller batch sizes

**Authentication**:
```python
# Check API key
import openai
client = openai.OpenAI(api_key="your_key")
client.models.list()  # Should work
```

**Quota Exceeded**:
- Check usage at [https://platform.openai.com/usage](https://platform.openai.com/usage)
- Upgrade plan or wait for quota reset

### Opik Not Working

**Connection Issues**:
```python
# Test Opik connection
from shared.utils import get_opik_client
client = get_opik_client()
print(f"Opik client: {client}")  # Should not be None
```

**Traces Not Showing**:
- Check `OPIK_ENABLED=true` in `.env`
- Verify API key is correct
- Check Opik dashboard project/workspace names match

---

## 10. Performance Comparison

| Metric | Sentence Transformers | OpenAI API |
|--------|----------------------|------------|
| **Cold start** | ~5-10s (model load) | ~0s (API) |
| **Throughput** | 100-500 texts/sec (GPU) | 1000+ texts/sec (API) |
| **Latency** | 10-50ms per text | 100-200ms per batch |
| **Cost** | $0 (compute cost) | $0.13/1M tokens |
| **Setup** | Complex (GPU drivers) | Simple (API key) |
| **Scaling** | Limited by hardware | Unlimited |

---

## 11. Next Steps

1. ✅ Update embedding generator to use OpenAI
2. ✅ Add Opik tracing decorators
3. ✅ Update configuration and environment variables
4. ⬜ Re-index all documents with new embeddings
5. ⬜ Set up Opik dashboard monitoring
6. ⬜ Configure cost alerts in OpenAI
7. ⬜ Update Airflow DAGs to use new embedding generator
8. ⬜ Add Opik tracking to RAG service (LangGraph)
9. ⬜ Create Opik dashboard for monitoring

---

## 12. Rollback Plan

If issues arise:

1. **Revert `embedding_generator.py`** to use Sentence Transformers
2. **Restore old dependencies** in `pyproject.toml`
3. **Update `.env`** to disable Opik
4. **Re-run** `uv sync`
5. **Keep old Qdrant collection** as backup during migration

**Backup Collection**:
```python
# Before migration, create backup
from qdrant_client import QdrantClient
client = QdrantClient("http://localhost:6333")
client.create_snapshot("eu_legal_documents_development")
```

---

**Migration completed successfully!** 🎉

For questions or issues, check:
- OpenAI API Status: [https://status.openai.com](https://status.openai.com)
- Opik Docs: [https://www.comet.com/docs/opik](https://www.comet.com/docs/opik)
- ConformAI GitHub Issues
