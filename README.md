# ConformAI

> RAG-based EU AI Compliance Intelligence System

ConformAI is a production-grade Retrieval-Augmented Generation (RAG) system designed to provide real-time, authoritative answers to questions about European Union AI regulations, including the EU AI Act, GDPR, and EDPB/EDPS guidelines.

## Features

- **Real-time Regulatory Awareness**: Continuously ingests the latest EU legal texts from official sources (EUR-Lex, EDPB, EDPS)
- **Legal-Grade RAG**: Article-level chunking, hybrid search (semantic + BM25), citation enforcement
- **LangGraph Orchestration**: Intelligent state machine for query analysis, retrieval, and grounded answer generation
- **Airflow Data Pipeline**: Automated ingestion, parsing, chunking, and indexing of legal documents
- **Microservices Architecture**: Scalable API Gateway, RAG Service, Retrieval Service, and Data Pipeline
- **Vector Search**: Qdrant-powered semantic search with metadata filtering
- **Production Ready**: Docker Compose for dev, Kubernetes manifests for production, monitoring with Prometheus

## Technology Stack

| Component | Technology |
|-----------|------------|
| **RAG Framework** | LangChain, LangGraph |
| **Vector Database** | Qdrant |
| **Pipeline Orchestration** | Apache Airflow |
| **API Framework** | FastAPI |
| **Database** | PostgreSQL |
| **Cache/Queue** | Redis, Celery |
| **LLM Providers** | Anthropic Claude, OpenAI |
| **Embeddings** | Sentence Transformers (BGE) |
| **Deployment** | Docker, Kubernetes |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- `uv` (fast Python package manager)

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ConformAI.git
cd ConformAI

# Create virtual environment (if not already created)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make dev

# Copy environment file
cp .env.example .env
# Edit .env and add your API keys (ANTHROPIC_API_KEY, etc.)
```

### 2. Start Infrastructure

```bash
# Start all services (Qdrant, PostgreSQL, Redis, Airflow, etc.)
make docker-up

# Initialize databases and collections
make init

# Initialize Airflow
make airflow-init
make airflow-user  # Creates admin/admin user
```

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Gateway** | http://localhost:8000/docs | - |
| **Airflow** | http://localhost:8080 | admin/admin |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | - |
| **MinIO Console** | http://localhost:9001 | minioadmin/minioadmin |

## Project Structure

```
ConformAI/
├── services/
│   ├── api-gateway/          # FastAPI gateway
│   ├── rag-service/          # LangGraph RAG orchestrator
│   ├── retrieval-service/    # LangChain retrieval
│   └── data-pipeline/        # Document processing
├── airflow/
│   ├── dags/                 # Airflow DAGs
│   └── plugins/              # Custom operators
├── shared/
│   ├── config/               # Shared configuration
│   ├── models/               # Pydantic models
│   └── utils/                # Utilities
├── data/
│   ├── raw/                  # Raw legal documents
│   ├── processed/            # Parsed & chunked
│   └── embeddings/           # Vector backups
├── scripts/                  # Initialization scripts
├── tests/                    # Unit & integration tests
└── infrastructure/           # Docker & K8s configs
```

## Development Workflow

### Running Services Locally

```bash
# Run API Gateway (with hot reload)
make run-api

# Or run individual services
cd services/api-gateway
uvicorn src.main:app --reload
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/unit/test_models.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint
```

### Airflow DAGs

```bash
# Trigger EUR-Lex ingestion DAG
docker-compose exec airflow-scheduler airflow dags trigger eurlex_daily_ingestion

# View DAG runs
make airflow-web
```

## API Usage

### Query Compliance Question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What obligations apply to high-risk AI systems under the EU AI Act?",
    "filters": {
      "regulations": ["AI Act"],
      "risk_category": "high"
    }
  }'
```

### Classify AI Use Case

```bash
curl -X POST "http://localhost:8000/api/v1/classify-usecase" \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Using facial recognition for employee attendance tracking"
  }'
```

## Data Pipeline

The system uses Apache Airflow to orchestrate data ingestion and processing:

### DAG 1: EUR-Lex Daily Ingestion
- **Schedule**: Daily at 2 AM
- **Tasks**: Fetch recent documents → Detect changes → Download XML/PDF → Store → Trigger processing

### DAG 2: Document Processing
- **Trigger**: On new documents
- **Tasks**: Parse legal documents → Extract hierarchy → Legal-aware chunking → Generate embeddings → Index to Qdrant

### DAG 3: GDPR & AI Act Monitoring
- **Schedule**: Weekly
- **Tasks**: Check for amendments → Fetch EDPB guidelines → Update vector store

## Configuration

Key environment variables in `.env`:

```bash
# LLM Provider
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022

# Vector Database
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Retrieval Settings
RETRIEVAL_TOP_K=10
RETRIEVAL_MIN_CONFIDENCE=0.6

# Data Sources
EURLEX_API_BASE_URL=https://publications.europa.eu/webapi/rdf/sparql
```

## Deployment

### Production with Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f infrastructure/k8s/

# Check deployment status
kubectl get pods -n conformai
```

### Scaling

```bash
# Scale RAG service
kubectl scale deployment rag-service --replicas=3 -n conformai
```

## Monitoring

- **Prometheus**: Metrics on `/metrics` endpoint
- **Grafana**: Dashboards for RAG performance, retrieval quality, data freshness
- **LangSmith**: LLM tracing (set `LANGCHAIN_TRACING_V2=true`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [Technical Implementation Plan](TECHNICAL_IMPLEMENTATION_PLAN.md)
- [Project Roadmap](claude.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Disclaimer

ConformAI provides informational support only and does not constitute legal advice. For legal guidance on EU AI compliance, consult a qualified legal professional.

## Acknowledgments

Built with:
- [LangChain](https://langchain.com) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Qdrant](https://qdrant.tech)
- [Apache Airflow](https://airflow.apache.org)
- [FastAPI](https://fastapi.tiangolo.com)
- EU official sources: [EUR-Lex](https://eur-lex.europa.eu), [EDPB](https://edpb.europa.eu)
