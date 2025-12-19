# Running the ConformAI Agent Test App

## Streamlit Interactive UI

### Prerequisites

1. **Environment Setup**
   ```bash
   # Make sure .env file is configured
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Install Dependencies**
   ```bash
   # Core dependencies
   uv pip install -e .

   # Streamlit
   uv pip install streamlit
   ```

3. **Start Required Services** (Optional - for full end-to-end testing)
   ```bash
   # Start Qdrant, PostgreSQL, Redis
   docker-compose up -d qdrant postgres redis

   # OR start all services
   docker-compose up -d
   ```

### Run the Streamlit App

```bash
# From project root
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features

#### 🔍 **Query Input**
- Enter custom queries or select from sample queries
- Samples include simple, medium, and complex queries

#### ⚙️ **Configuration**
- **Max Iterations**: Control ReAct loop iterations (1-10)
- **Display Options**: Toggle visibility of reasoning, sub-queries, retrievals, raw state

#### 📊 **Real-Time Visualization**

1. **Query Analysis**
   - Intent detection
   - Complexity classification
   - AI domain identification
   - Risk category

2. **Query Decomposition**
   - View how complex queries are broken into sub-questions
   - See priority levels and aspects
   - Track completion status

3. **Agent Reasoning (ReAct Loop)**
   - Step-by-step agent thoughts
   - Actions taken at each step
   - Observations and results
   - See how the agent plans, acts, and observes

4. **Retrieval History**
   - All retrieval calls made by the agent
   - Number of chunks retrieved
   - Search queries used

5. **Final Answer**
   - Complete answer with citations
   - Formatted markdown
   - Legal disclaimer

6. **Citations**
   - All source references
   - Regulation names and article numbers
   - Excerpts from sources

7. **Performance Metrics**
   - Processing time
   - Confidence score
   - Number of iterations
   - Chunks retrieved
   - Grounding validation status

8. **Query History**
   - Last 5 queries saved in session
   - Quick access to previous results

### Sample Queries to Test

#### Simple Query
```
What is a high-risk AI system?
```
- Expected: 1 iteration, direct answer
- Processing: ~1-2 seconds

#### Medium Complexity
```
What are the documentation requirements for high-risk AI systems?
```
- Expected: 2-3 iterations
- May decompose into sub-questions
- Processing: ~2-3 seconds

#### Complex Multi-Aspect
```
What are the documentation requirements for recruitment AI vs healthcare AI systems, and how do they differ?
```
- Expected: 4-5 iterations
- Decomposes into 3-4 sub-questions
- Agent retrieves multiple times
- Synthesizes comparative answer
- Processing: ~3-5 seconds

#### Comparative Analysis
```
Compare the obligations, prohibitions, and transparency requirements for biometric identification systems versus AI in recruitment.
```
- Expected: 5+ iterations
- Multiple sub-questions
- Complex synthesis
- Processing: ~4-6 seconds

### Troubleshooting

#### "Settings error" in sidebar
- Make sure `.env` file exists and has required variables
- Check `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set

#### "Error processing query"
- Check that services are running (if using Retrieval Service)
- For testing without services, the RAG will fall back to mock/direct mode
- Check logs for specific errors

#### Slow processing
- First query may be slower (model loading)
- Complex queries with 5 iterations can take 4-6 seconds
- Reduce `max_iterations` for faster testing

### Tips for Testing

1. **Start Simple**: Test with simple queries first to verify the pipeline works
2. **Increase Complexity**: Gradually test more complex queries
3. **Watch Agent Reasoning**: Enable "Show Agent Reasoning" to understand how the agent thinks
4. **Compare Iterations**: Try the same query with different max_iterations values
5. **Test Refusals**: Try out-of-scope queries to see safety guardrails

---

## Command-Line Testing

For quick command-line testing without UI:

```bash
python test_rag.py
```

This runs 3 test queries of increasing complexity and shows the results in the terminal.

### Modify Test Queries

Edit `test_rag.py` and change the `test_queries` list:

```python
test_queries = [
    "Your custom query here",
    "Another query",
]
```

---

## Running Individual Services

### RAG Service Only

```bash
cd services/rag-service
python src/api/main.py
```

Access at `http://localhost:8001`

Test with curl:
```bash
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are high-risk AI systems?",
    "max_iterations": 5
  }'
```

### Retrieval Service Only

```bash
cd services/retrieval-service
python src/api/main.py
```

Access at `http://localhost:8002`

Test with curl:
```bash
curl -X POST http://localhost:8002/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "high-risk AI obligations",
    "top_k": 10
  }'
```

---

## Full Stack Testing

1. **Start all infrastructure**:
   ```bash
   docker-compose up -d
   ```

2. **Run data pipeline** (one-time):
   ```bash
   # Trigger Airflow DAGs to ingest and process documents
   # Access Airflow UI at http://localhost:8080
   ```

3. **Start RAG service**:
   ```bash
   cd services/rag-service && python src/api/main.py
   ```

4. **Start Retrieval service**:
   ```bash
   cd services/retrieval-service && python src/api/main.py
   ```

5. **Run Streamlit app**:
   ```bash
   streamlit run app.py
   ```

Now you have the full stack running and can test end-to-end!

---

## Development Mode

For rapid iteration during development:

1. **Use Haiku for faster testing**:
   Edit `.env`:
   ```bash
   LLM_PROVIDER=anthropic
   LLM_MODEL=claude-3-5-haiku-20241022
   ```

2. **Reduce max iterations**:
   Set to 2-3 for quick testing

3. **Enable debug mode**:
   In Streamlit, check "Show Raw State (Debug)"

---

## Next Steps

- ✅ Test simple queries
- ✅ Test complex queries with decomposition
- ✅ Verify agent reasoning is logical
- ✅ Check citation accuracy
- ✅ Test safety refusals (out-of-scope queries)
- ✅ Monitor performance metrics
- ✅ Compare different models (GPT-4o vs Claude Sonnet)

Happy testing! 🚀
