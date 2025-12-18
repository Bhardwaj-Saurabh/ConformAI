# Getting Started with ConformAI

## What's Been Set Up

Congratulations! Your ConformAI project has been initialized with:

### ✅ Project Structure
- **Complete microservices architecture** with 4 services:
  - `api-gateway/` - FastAPI REST API
  - `rag-service/` - LangGraph RAG orchestrator
  - `retrieval-service/` - LangChain retrieval layer
  - `data-pipeline/` - Document processing (integrated with Airflow)

- **Airflow** data pipeline with EUR-Lex ingestion DAG
- **Shared modules** for config, models, and utilities
- **Infrastructure** configs (Docker Compose, K8s templates)

### ✅ Dependencies Installed
All 259+ packages installed via `uv`, including:
- LangChain ecosystem (LangChain, LangGraph, LangSmith)
- Vector database (Qdrant client)
- ML libraries (Sentence Transformers, Transformers, PyTorch)
- Airflow with providers
- FastAPI with async support
- Database tools (SQLAlchemy, PostgreSQL, Redis)

### ✅ Configuration Files
- `pyproject.toml` - Complete dependency manifest
- `docker-compose.yml` - 8 containerized services
- `Makefile` - Development commands
- `.env` - Environment configuration (needs your API keys)
- `README.md` - Full documentation

---

## Next Steps

### 1. Configure API Keys

Edit the `.env` file and add your API keys:

```bash
# Required
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional (if using OpenAI)
OPENAI_API_KEY=your_openai_key_here

# Optional (for LLM tracing)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
```

### 2. Start Infrastructure Services

```bash
# Start Qdrant, PostgreSQL, Redis, MinIO
make docker-up

# Wait ~30 seconds for services to be healthy
docker-compose ps
```

### 3. Initialize Databases

```bash
# Initialize Qdrant collections and PostgreSQL
python scripts/init_project.py

# Initialize Airflow database
make airflow-init

# Create Airflow admin user (username: admin, password: admin)
make airflow-user
```

### 4. Test the API Gateway

```bash
# Run API Gateway locally (with hot reload)
make run-api

# Or use Docker
docker-compose up -d api-gateway

# Visit API docs
open http://localhost:8000/docs
```

### 5. Access Services

| Service | URL | Login |
|---------|-----|-------|
| **API Docs** | http://localhost:8000/docs | - |
| **Airflow UI** | http://localhost:8080 | admin/admin |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | - |
| **MinIO Console** | http://localhost:9001 | minioadmin/minioadmin |

---

## Development Workflow

### Running Airflow DAGs

```bash
# Trigger the EUR-Lex ingestion DAG
docker-compose exec airflow-scheduler \
  airflow dags trigger eurlex_daily_ingestion

# View DAG status in Airflow UI
make airflow-web
```

### Testing the RAG Pipeline

Once you've implemented the RAG service (see TECHNICAL_IMPLEMENTATION_PLAN.md):

```bash
# Query the compliance API
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the obligations for high-risk AI systems?",
    "filters": {"regulations": ["AI Act"]}
  }'
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Run tests (once implemented)
make test
```

---

## What to Build Next

Follow the [TECHNICAL_IMPLEMENTATION_PLAN.md](TECHNICAL_IMPLEMENTATION_PLAN.md) for step-by-step implementation:

### Phase 1: Data Pipeline (Recommended Start)
1. **Implement EUR-Lex API integration** in `airflow/dags/eurlex_ingestion_dag.py`
2. **Build document parser** in `services/data-pipeline/src/parsers/`
3. **Create legal-aware chunker**
4. **Generate embeddings** using Sentence Transformers
5. **Index to Qdrant**

### Phase 2: Retrieval Service
1. **Setup Qdrant collections** with legal metadata schema
2. **Implement hybrid retrieval** (semantic + BM25)
3. **Build query classifier**
4. **Add reranking layer**

### Phase 3: RAG Orchestrator (LangGraph)
1. **Create state machine** for RAG workflow
2. **Implement nodes** (analyze, retrieve, generate, validate)
3. **Add citation extraction**
4. **Build grounding validation**

### Phase 4: API Endpoints
1. **Implement `/api/v1/query`** endpoint
2. **Add `/api/v1/classify-usecase`**
3. **Create `/api/v1/compliance-checklist`**
4. **Add authentication & rate limiting**

---

## Quick Commands Reference

```bash
# Development
make install          # Install dependencies
make dev             # Install with dev dependencies
make run-api         # Run API Gateway locally

# Docker
make docker-up       # Start all services
make docker-down     # Stop all services
make docker-logs     # View logs
make docker-rebuild  # Rebuild containers

# Airflow
make airflow-init    # Initialize Airflow DB
make airflow-user    # Create admin user
make airflow-web     # Open Airflow UI

# Database
make db-migrate      # Create migration
make db-upgrade      # Apply migrations

# Testing & Quality
make test           # Run tests
make lint           # Run linters
make format         # Format code
make clean          # Clean build artifacts

# Full setup (new developers)
make quickstart     # Does everything: install, init, docker-up, airflow setup
```

---

## Project Files Overview

| File/Directory | Purpose |
|----------------|---------|
| `shared/` | Common code shared across all services |
| `shared/config/settings.py` | Centralized configuration management |
| `shared/models/legal_document.py` | Pydantic models for legal documents |
| `services/api-gateway/src/main.py` | FastAPI application entry point |
| `airflow/dags/eurlex_ingestion_dag.py` | EUR-Lex data ingestion pipeline |
| `docker-compose.yml` | All services orchestration |
| `Makefile` | Development commands |
| `.env` | Environment variables (add your API keys here) |

---

## Architecture Highlights

### Microservices
- **API Gateway** (Port 8000) - Public REST API
- **RAG Service** (Port 8001) - LangGraph orchestrator
- **Retrieval Service** (Port 8002) - LangChain retrieval

### Infrastructure
- **Qdrant** (Port 6333) - Vector database for embeddings
- **PostgreSQL** (Port 5432) - Metadata & versions
- **Redis** (Port 6379) - Cache & Celery broker
- **Airflow** (Port 8080) - Data pipeline orchestration
- **MinIO** (Ports 9000/9001) - S3-compatible object storage

### Data Flow
```
EUR-Lex API
    ↓
Airflow DAG (ingest)
    ↓
Document Parser
    ↓
Legal Chunker
    ↓
Embeddings (BGE)
    ↓
Qdrant (vector store)
    ↓
LangGraph RAG
    ↓
FastAPI Response
```

---

## Troubleshooting

### Services won't start
```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs qdrant
docker-compose logs postgres

# Restart specific service
docker-compose restart qdrant
```

### Airflow shows "No module named 'shared'"
```bash
# Rebuild Airflow with shared modules
docker-compose build airflow-webserver
docker-compose up -d airflow-webserver airflow-scheduler
```

### Can't connect to Qdrant
```bash
# Check Qdrant health
curl http://localhost:6333/

# Restart Qdrant
docker-compose restart qdrant
```

---

## Resources

- **Technical Plan**: [TECHNICAL_IMPLEMENTATION_PLAN.md](TECHNICAL_IMPLEMENTATION_PLAN.md)
- **Project Vision**: [claude.md](claude.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **LangChain Docs**: https://python.langchain.com/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Qdrant Docs**: https://qdrant.tech/documentation/
- **Airflow Docs**: https://airflow.apache.org/docs/

---

## Support

For questions or issues:
1. Check [TECHNICAL_IMPLEMENTATION_PLAN.md](TECHNICAL_IMPLEMENTATION_PLAN.md)
2. Review service logs: `docker-compose logs <service-name>`
3. Verify .env configuration
4. Check service health: `docker-compose ps`

---

**Ready to build production-grade legal RAG!** 🚀

Next recommended action: **Start Docker services** with `make docker-up`
