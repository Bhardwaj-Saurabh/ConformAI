# ConformAI - Technical Implementation Plan
## RAG-Based EU AI Compliance System using LangChain, LangGraph & Airflow

---

## Technology Stack Overview

### Core Framework
- **LangChain**: Retrieval, embeddings, document processing, LLM integration
- **LangGraph**: State machine orchestration for RAG workflows
- **Apache Airflow**: Data pipeline orchestration and scheduling

### Supporting Infrastructure
- **Vector Database**: Qdrant or Weaviate (legal-domain optimized)
- **Metadata Store**: PostgreSQL
- **Object Storage**: MinIO (S3-compatible) or AWS S3
- **Message Queue**: Redis/Celery for async tasks
- **API Framework**: FastAPI
- **Containerization**: Docker + Kubernetes
- **Monitoring**: Prometheus + Grafana + OpenTelemetry

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway (FastAPI)                    │
│                    - Rate limiting                           │
│                    - Authentication                          │
│                    - Request routing                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌────────────────────┐      ┌────────────────────┐
│  LangGraph RAG     │      │  Data Pipeline     │
│  Orchestrator      │      │  (Airflow)         │
│                    │      │                    │
│  - Query Analysis  │      │  - EUR-Lex API     │
│  - Retrieval       │      │  - Document Parse  │
│  - Generation      │      │  - Chunking        │
│  - Citation        │      │  - Embedding       │
└─────────┬──────────┘      └─────────┬──────────┘
          │                           │
          ▼                           ▼
┌────────────────────┐      ┌────────────────────┐
│  LangChain         │      │  Vector DB         │
│  Retrieval Service │◄─────┤  (Qdrant)          │
│                    │      │                    │
│  - Hybrid Search   │      │  - Embeddings      │
│  - Reranking       │      │  - Metadata        │
└────────────────────┘      └────────────────────┘
          │
          ▼
┌────────────────────┐
│  PostgreSQL        │
│  - Document meta   │
│  - Versions        │
│  - Audit logs      │
└────────────────────┘
```

---

## Phase 1: Project Setup & Infrastructure

### 1.1 Project Structure
```
ConformAI/
├── services/
│   ├── api-gateway/          # FastAPI gateway
│   ├── rag-service/          # LangGraph RAG orchestrator
│   ├── retrieval-service/    # LangChain retrieval
│   ├── data-pipeline/        # Airflow DAGs
│   └── shared/               # Common utilities
├── infrastructure/
│   ├── docker/               # Dockerfiles
│   ├── k8s/                  # Kubernetes manifests
│   └── terraform/            # IaC (optional)
├── airflow/
│   ├── dags/                 # Airflow DAGs
│   ├── plugins/              # Custom operators
│   └── config/               # Airflow configs
├── data/
│   ├── raw/                  # Raw documents
│   ├── processed/            # Parsed/chunked
│   └── embeddings/           # Vector store backups
├── tests/
├── docs/
├── pyproject.toml
├── docker-compose.yml
└── README.md
```

### 1.2 Core Dependencies (pyproject.toml)
```toml
[project]
name = "conformai"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # LangChain ecosystem
    "langchain>=0.1.0",
    "langchain-community>=0.1.0",
    "langchain-core>=0.1.0",
    "langgraph>=0.1.0",
    "langsmith>=0.1.0",

    # LLM providers
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",

    # Vector stores
    "qdrant-client>=1.7.0",
    "langchain-qdrant>=0.1.0",

    # Embeddings
    "sentence-transformers>=2.2.0",
    "transformers>=4.35.0",

    # Document processing
    "beautifulsoup4>=4.12.0",
    "lxml>=4.9.0",
    "pypdf>=3.17.0",
    "python-docx>=1.1.0",
    "unstructured>=0.11.0",

    # Airflow
    "apache-airflow>=2.8.0",
    "apache-airflow-providers-http>=4.7.0",
    "apache-airflow-providers-postgres>=5.9.0",

    # API & Web
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "httpx>=0.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",

    # Database
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "alembic>=1.13.0",

    # Utilities
    "python-dotenv>=1.0.0",
    "redis>=5.0.0",
    "celery>=5.3.0",
    "tenacity>=8.2.0",

    # Monitoring
    "prometheus-client>=0.19.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",

    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
]
```

---

## Phase 2: Data Pipeline with Airflow

### 2.1 Airflow DAG Architecture

**DAG 1: EUR-Lex Ingestion** (Daily)
```python
# airflow/dags/eurlex_ingestion_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'conformai',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'eurlex_daily_ingestion',
    default_args=default_args,
    description='Fetch latest EU regulations from EUR-Lex',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['data-ingestion', 'eurlex'],
)

# Tasks:
# 1. fetch_recent_documents (EUR-Lex API)
# 2. detect_changes (compare versions)
# 3. download_documents (XML/PDF)
# 4. store_raw_documents (S3/MinIO)
# 5. trigger_processing_dag
```

**DAG 2: GDPR & AI Act Monitoring** (Weekly)
```python
# airflow/dags/gdpr_aiact_monitoring_dag.py

# Tasks:
# 1. check_gdpr_amendments
# 2. check_aiact_updates
# 3. fetch_edpb_guidelines
# 4. fetch_edps_opinions
# 5. store_and_trigger_processing
```

**DAG 3: Document Processing Pipeline** (Triggered)
```python
# airflow/dags/document_processing_dag.py

# Tasks:
# 1. parse_legal_documents (XML/PDF → structured)
# 2. extract_legal_hierarchy (Regulation/Chapter/Article)
# 3. legal_aware_chunking (preserve article boundaries)
# 4. attach_metadata (regulation, article, date, domain)
# 5. generate_embeddings (sentence-transformers)
# 6. index_to_vectordb (Qdrant)
# 7. update_metadata_store (PostgreSQL)
# 8. validate_indexing
```

### 2.2 Data Source Operators

**EUR-Lex API Operator**
```python
# airflow/plugins/operators/eurlex_operator.py

class EURLexFetchOperator(BaseOperator):
    """
    Custom operator to fetch documents from EUR-Lex SPARQL endpoint

    Features:
    - Query by CELEX number
    - Filter by document type (regulation, directive)
    - Get latest versions
    - Download XML/PDF formats
    """

    def __init__(
        self,
        celex_pattern: str,
        document_types: list[str],
        start_date: str,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.celex_pattern = celex_pattern
        self.document_types = document_types
        self.start_date = start_date

    def execute(self, context):
        # Implementation with EUR-Lex SPARQL queries
        pass
```

**EDPB Guidelines Scraper**
```python
# airflow/plugins/operators/edpb_operator.py

class EDPBGuidelineOperator(BaseOperator):
    """
    Scrape EDPB guidelines and opinions

    Sources:
    - https://edpb.europa.eu/our-work-tools/our-documents/guidelines_en
    - RSS feeds for updates
    """
    pass
```

### 2.3 Document Processing Tasks

**Legal Document Parser**
```python
# services/data-pipeline/parsers/legal_parser.py

from langchain.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EULegalDocumentParser:
    """
    Parse EU legal documents preserving hierarchy

    Handles:
    - Formex XML format (EUR-Lex standard)
    - PDF with OCR fallback
    - HTML from EDPB
    """

    def parse_regulation(self, doc_path: str) -> LegalDocument:
        """
        Extract:
        - CELEX number
        - Title
        - Date of adoption
        - Chapters → Articles → Paragraphs
        - Annexes
        - References to other regulations
        """
        pass

    def extract_hierarchy(self, xml_tree) -> dict:
        """
        Build tree structure:
        {
            "regulation": "2016/679",
            "chapters": [
                {
                    "number": "I",
                    "title": "General Provisions",
                    "articles": [...]
                }
            ]
        }
        """
        pass
```

**Legal-Aware Chunking Strategy**
```python
# services/data-pipeline/chunking/legal_chunker.py

from langchain.text_splitter import TextSplitter

class LegalArticleChunker(TextSplitter):
    """
    Chunk legal documents respecting article boundaries

    Strategy:
    1. Primary chunks: Full articles
    2. If article > max_tokens: Split by paragraph
    3. Never split mid-paragraph
    4. Preserve article context in metadata
    """

    def __init__(
        self,
        max_chunk_size: int = 512,  # tokens
        overlap_sentences: int = 2,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def split_article(self, article: Article) -> list[Chunk]:
        """
        Returns chunks with metadata:
        {
            "text": "Article 22...",
            "metadata": {
                "regulation": "GDPR",
                "celex": "32016R0679",
                "article_number": "22",
                "article_title": "Automated decision-making",
                "chunk_index": 0,
                "total_chunks": 2,
                "effective_date": "2018-05-25",
                "domains": ["automated_decision_making", "ai"],
                "risk_category": "high",
            }
        }
        """
        pass
```

### 2.4 Embedding Generation

```python
# services/data-pipeline/embeddings/legal_embeddings.py

from langchain_openai import OpenAIEmbeddings

class LegalEmbeddingGenerator:
    """
    Generate embeddings optimized for legal text

    Using OpenAI embeddings via API:
    - text-embedding-3-large (1024-3072 dimensions)
    - text-embedding-3-small (1536 dimensions)
    """

    def __init__(self, model_name: str = "text-embedding-3-large", dimensions: int = 1024):
        from langchain_openai import OpenAIEmbeddings

        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            dimensions=dimensions  # Adjustable from 256-3072
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """
        Generate embeddings with:
        - Dense vector for semantic search
        - Sparse keywords for BM25
        - Metadata for filtering
        """
        pass
```

---

## Phase 3: Vector Database Setup (Qdrant)

### 3.1 Collection Schema

```python
# services/retrieval-service/vectordb/qdrant_setup.py

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

def create_legal_collection():
    client = QdrantClient(url="http://qdrant:6333")

    client.create_collection(
        collection_name="eu_legal_documents",
        vectors_config=VectorParams(
            size=1024,  # BGE-large dimension
            distance=Distance.COSINE
        ),
        # Payload indexing for fast filtering
        payload_schema={
            "regulation": PayloadSchemaType.KEYWORD,
            "celex": PayloadSchemaType.KEYWORD,
            "article_number": PayloadSchemaType.KEYWORD,
            "effective_date": PayloadSchemaType.DATETIME,
            "domains": PayloadSchemaType.KEYWORD,
            "risk_category": PayloadSchemaType.KEYWORD,
            "version": PayloadSchemaType.KEYWORD,
        }
    )

    # Create separate collection for EDPB guidelines
    client.create_collection(
        collection_name="edpb_guidelines",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
```

### 3.2 Indexing Pipeline

```python
# services/data-pipeline/indexing/vectordb_indexer.py

from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings

class VectorDBIndexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024
        )
        self.vectorstore = Qdrant(
            client=QdrantClient(url="http://qdrant:6333"),
            collection_name="eu_legal_documents",
            embeddings=self.embeddings
        )

    def index_chunks(self, chunks: list[Chunk]):
        """
        Batch index with metadata
        """
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas
        )

    def update_document_version(
        self,
        celex: str,
        old_version: str,
        new_version: str
    ):
        """
        Handle document updates:
        1. Mark old version as deprecated
        2. Index new version
        3. Maintain version history
        """
        pass
```

---

## Phase 4: LangChain Retrieval Service

### 4.1 Hybrid Retrieval Strategy

```python
# services/retrieval-service/retrievers/hybrid_retriever.py

from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import Qdrant
from langchain.retrievers import BM25Retriever

class LegalHybridRetriever:
    """
    Combine semantic + keyword search for legal queries
    """

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        # Dense retriever (vector similarity)
        self.dense_retriever = Qdrant(
            client=QdrantClient(url="http://qdrant:6333"),
            collection_name="eu_legal_documents",
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-large",
                dimensions=1024
            )
        ).as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance
            search_kwargs={
                "k": 10,
                "fetch_k": 50,
                "lambda_mult": 0.7  # Diversity vs relevance
            }
        )

        # Sparse retriever (BM25 keyword)
        self.sparse_retriever = BM25Retriever.from_documents(
            documents=self._load_all_documents()
        )

        # Ensemble
        self.ensemble = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.7, 0.3]  # Favor semantic search
        )

    def retrieve_with_filters(
        self,
        query: str,
        filters: dict
    ) -> list[Document]:
        """
        Apply metadata filters:
        - regulation: ["GDPR", "AI Act"]
        - domains: ["recruitment", "biometrics"]
        - risk_category: "high"
        - effective_after: "2024-01-01"
        """
        filter_conditions = self._build_qdrant_filter(filters)

        docs = self.dense_retriever.get_relevant_documents(
            query=query,
            filter=filter_conditions
        )

        return docs
```

### 4.2 Query Analysis & Classification

```python
# services/retrieval-service/query_analysis/classifier.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class QueryClassifier:
    """
    Classify user queries to optimize retrieval
    """

    def __init__(self, llm):
        self.llm = llm
        self.classification_chain = self._build_chain()

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify into:
        1. AI Domain: recruitment, biometrics, healthcare, surveillance, etc.
        2. Risk Category: prohibited, high-risk, limited-risk, minimal-risk
        3. Regulation Scope: GDPR, AI Act, both
        4. Question Type: compliance, obligation, prohibition, definition

        Example:
        Query: "Can I use facial recognition for employee monitoring?"
        Output: {
            "domain": "biometrics",
            "subdomain": "workplace_surveillance",
            "risk_category": "high",
            "regulations": ["GDPR", "AI Act"],
            "question_type": "prohibition_check"
        }
        """
        result = self.classification_chain.run(query=query)
        return QueryClassification.parse(result)

    def extract_legal_references(self, query: str) -> list[str]:
        """
        Extract explicit legal references:
        - "Article 22 GDPR" → ["GDPR:Article 22"]
        - "high-risk AI system" → ["AI Act:Article 6"]
        """
        pass
```

### 4.3 Reranking Layer

```python
# services/retrieval-service/reranking/legal_reranker.py

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers import CrossEncoder

class LegalReranker:
    """
    Rerank retrieved documents for legal relevance
    """

    def __init__(self, base_retriever):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        self.compressor = CrossEncoderReranker(
            model=self.cross_encoder,
            top_n=5
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )

    def rerank_with_legal_priority(
        self,
        query: str,
        documents: list[Document]
    ) -> list[Document]:
        """
        Rerank considering:
        1. Semantic relevance (cross-encoder)
        2. Recency (newer regulations prioritized)
        3. Legal hierarchy (regulations > guidelines > opinions)
        4. Specificity (specific articles > general chapters)
        """
        pass
```

---

## Phase 5: LangGraph RAG Orchestrator

### 5.1 State Machine Definition

```python
# services/rag-service/graph/rag_graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage

class RAGState(TypedDict):
    """State for RAG workflow"""
    query: str
    query_classification: dict
    retrieved_documents: list[Document]
    reranked_documents: list[Document]
    generated_answer: str
    citations: list[dict]
    confidence_score: float
    should_refuse: bool
    error: str | None

def create_rag_graph():
    """
    LangGraph workflow:

    START
      ↓
    analyze_query ──────────┐
      ↓                     │
    classify_usecase        │
      ↓                     │
    retrieve_documents      │
      ↓                     │
    apply_filters           │
      ↓                     │
    rerank_results          │
      ↓                     │
    check_confidence ───────┤→ [low] → refuse_to_answer → END
      ↓ [high]              │
    generate_answer         │
      ↓                     │
    extract_citations       │
      ↓                     │
    validate_grounding ─────┤→ [fail] → retry_generation
      ↓ [pass]              │
    format_response         │
      ↓                     │
    END                     │
    """

    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("classify_usecase", classify_usecase_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("rerank_results", rerank_results_node)
    workflow.add_node("check_confidence", check_confidence_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("extract_citations", extract_citations_node)
    workflow.add_node("validate_grounding", validate_grounding_node)
    workflow.add_node("refuse_to_answer", refuse_to_answer_node)
    workflow.add_node("format_response", format_response_node)

    # Add edges
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "classify_usecase")
    workflow.add_edge("classify_usecase", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "rerank_results")
    workflow.add_edge("rerank_results", "check_confidence")

    # Conditional routing based on confidence
    workflow.add_conditional_edges(
        "check_confidence",
        lambda state: "generate" if state["confidence_score"] > 0.6 else "refuse",
        {
            "generate": "generate_answer",
            "refuse": "refuse_to_answer"
        }
    )

    workflow.add_edge("generate_answer", "extract_citations")
    workflow.add_edge("extract_citations", "validate_grounding")

    # Conditional: retry if grounding fails
    workflow.add_conditional_edges(
        "validate_grounding",
        lambda state: "format" if state.get("grounding_valid") else "generate_answer",
        {
            "format": "format_response",
            "generate_answer": "generate_answer"  # Retry
        }
    )

    workflow.add_edge("refuse_to_answer", END)
    workflow.add_edge("format_response", END)

    return workflow.compile()
```

### 5.2 Node Implementations

```python
# services/rag-service/graph/nodes.py

from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

def analyze_query_node(state: RAGState) -> RAGState:
    """
    Parse user query:
    - Extract intent
    - Identify explicit legal references
    - Detect temporal scope (current law vs historical)
    """
    query = state["query"]

    # Use LLM to analyze query structure
    analyzer = QueryAnalyzer(llm=ChatAnthropic(model="claude-3-5-sonnet-20241022"))
    analysis = analyzer.analyze(query)

    state["query_classification"] = analysis
    return state

def retrieve_documents_node(state: RAGState) -> RAGState:
    """
    Retrieve from vector DB with filters
    """
    classification = state["query_classification"]

    retriever = LegalHybridRetriever()
    docs = retriever.retrieve_with_filters(
        query=state["query"],
        filters={
            "domains": classification.get("domains", []),
            "regulations": classification.get("regulations", [])
        }
    )

    state["retrieved_documents"] = docs
    return state

def generate_answer_node(state: RAGState) -> RAGState:
    """
    Generate grounded answer with citations
    """
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0  # Deterministic for legal
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_RAG_SYSTEM_PROMPT),
        ("user", LEGAL_RAG_USER_TEMPLATE)
    ])

    chain = prompt | llm

    response = chain.invoke({
        "query": state["query"],
        "context": state["reranked_documents"],
        "classification": state["query_classification"]
    })

    state["generated_answer"] = response.content
    return state

def validate_grounding_node(state: RAGState) -> RAGState:
    """
    Ensure every claim is backed by retrieved documents
    """
    validator = GroundingValidator()
    is_valid = validator.validate(
        answer=state["generated_answer"],
        sources=state["reranked_documents"],
        citations=state["citations"]
    )

    state["grounding_valid"] = is_valid
    return state

def refuse_to_answer_node(state: RAGState) -> RAGState:
    """
    Handle low-confidence scenarios
    """
    state["generated_answer"] = (
        "I don't have sufficient information in the current EU legal sources "
        "to provide a reliable answer to this question. This could be because:\n"
        "- The topic is not yet regulated\n"
        "- The regulation is pending implementation\n"
        "- The query requires case-specific legal interpretation\n\n"
        "**Disclaimer**: This system provides informational support only. "
        "For legal advice, consult a qualified EU law professional."
    )
    state["should_refuse"] = True
    return state
```

### 5.3 Prompt Templates

```python
# services/rag-service/prompts/legal_prompts.py

LEGAL_RAG_SYSTEM_PROMPT = """You are a specialized legal assistant for EU AI and data protection compliance.

**Core Principles:**
1. **Grounded Only**: Base all answers EXCLUSIVELY on the provided legal sources
2. **Explicit Citations**: Cite specific articles, paragraphs, and regulations
3. **No Speculation**: If information is missing, state it clearly
4. **Disclaimers**: Remind users this is informational, not legal advice

**Answer Structure:**
1. **Summary**: Brief answer to the question
2. **Legal Basis**: Relevant articles and provisions
3. **Obligations**: What must be done
4. **Prohibitions**: What is forbidden
5. **Compliance Checklist**: Actionable steps
6. **Citations**: Numbered references to source documents

**Tone**: Professional, precise, objective

**Sources Provided Below:**
"""

LEGAL_RAG_USER_TEMPLATE = """**User Question:**
{query}

**Query Classification:**
- Domain: {classification[domain]}
- Risk Category: {classification[risk_category]}
- Relevant Regulations: {classification[regulations]}

**Retrieved Legal Sources:**
{context}

**Instructions:**
Answer the user's question following the answer structure defined in the system prompt.
Ensure every factual claim includes a citation in the format [Source N: Article X].
If the sources don't contain sufficient information, acknowledge the limitation.

**Answer:**
"""
```

---

## Phase 6: FastAPI Gateway

### 6.1 API Endpoints

```python
# services/api-gateway/main.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

app = FastAPI(
    title="ConformAI API",
    description="EU AI Compliance Intelligence",
    version="0.1.0"
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

class ComplianceQuery(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    context: dict | None = Field(
        default=None,
        description="Additional context: industry, company_size, etc."
    )
    filters: dict | None = Field(
        default=None,
        description="Filters: regulations, domains, risk_category"
    )

class ComplianceResponse(BaseModel):
    answer: str
    summary: str
    legal_basis: list[dict]
    obligations: list[str]
    prohibitions: list[str]
    compliance_checklist: list[dict]
    citations: list[dict]
    confidence_score: float
    disclaimer: str
    metadata: dict

@app.post("/api/v1/query", response_model=ComplianceResponse)
@limiter.limit("10/minute")
async def query_compliance(
    request: ComplianceQuery,
    api_key: str = Depends(verify_api_key)
):
    """
    Main RAG endpoint for compliance questions
    """
    try:
        rag_graph = get_rag_graph()

        result = await rag_graph.ainvoke({
            "query": request.query,
            "filters": request.filters or {}
        })

        return ComplianceResponse(
            answer=result["generated_answer"],
            summary=result["summary"],
            legal_basis=result["legal_basis"],
            obligations=result["obligations"],
            prohibitions=result["prohibitions"],
            compliance_checklist=result["checklist"],
            citations=result["citations"],
            confidence_score=result["confidence_score"],
            disclaimer=LEGAL_DISCLAIMER,
            metadata={
                "query_classification": result["query_classification"],
                "sources_count": len(result["reranked_documents"]),
                "processing_time_ms": result["processing_time"]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/classify-usecase")
@limiter.limit("20/minute")
async def classify_ai_usecase(
    description: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Classify AI use case into EU AI Act risk categories
    """
    classifier = QueryClassifier(llm=get_llm())
    classification = classifier.classify_query(description)

    return {
        "domain": classification.domain,
        "risk_category": classification.risk_category,
        "relevant_regulations": classification.regulations,
        "key_articles": classification.key_articles,
        "compliance_complexity": classification.complexity_score
    }

@app.get("/api/v1/regulations/{celex_id}")
async def get_regulation_details(celex_id: str):
    """
    Get full details of a specific regulation
    """
    pass

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_db": await check_qdrant_health(),
        "postgres": await check_postgres_health(),
        "airflow": await check_airflow_health()
    }
```

---

## Phase 7: Deployment & Orchestration

### 7.1 Docker Compose (Development)

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}

  # PostgreSQL
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: conformai
      POSTGRES_USER: conformai
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  # Redis (for Celery/caching)
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # Airflow (webserver + scheduler)
  airflow-webserver:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    depends_on:
      - postgres
      - redis
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://conformai:${POSTGRES_PASSWORD}@postgres/conformai
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./data:/opt/airflow/data
    ports:
      - "8080:8080"
    command: webserver

  airflow-scheduler:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    depends_on:
      - postgres
      - redis
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://conformai:${POSTGRES_PASSWORD}@postgres/conformai
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./data:/opt/airflow/data
    command: scheduler

  airflow-worker:
    build:
      context: ./airflow
      dockerfile: Dockerfile
    depends_on:
      - postgres
      - redis
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://conformai:${POSTGRES_PASSWORD}@postgres/conformai
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
      - ./data:/opt/airflow/data
    command: celery worker

  # RAG Service
  rag-service:
    build:
      context: ./services/rag-service
      dockerfile: Dockerfile
    depends_on:
      - qdrant
      - postgres
    environment:
      - QDRANT_URL=http://qdrant:6333
      - POSTGRES_URL=postgresql://conformai:${POSTGRES_PASSWORD}@postgres/conformai
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    ports:
      - "8001:8000"

  # API Gateway
  api-gateway:
    build:
      context: ./services/api-gateway
      dockerfile: Dockerfile
    depends_on:
      - rag-service
    environment:
      - RAG_SERVICE_URL=http://rag-service:8000
    ports:
      - "8000:8000"

volumes:
  qdrant_storage:
  postgres_data:
```

### 7.2 Kubernetes Deployment (Production)

```yaml
# infrastructure/k8s/qdrant-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 3
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: qdrant-storage
        persistentVolumeClaim:
          claimName: qdrant-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
  type: ClusterIP
```

---

## Phase 8: Monitoring & Evaluation

### 8.1 RAG Evaluation Metrics

```python
# services/rag-service/evaluation/metrics.py

from langchain.evaluation import load_evaluator
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

class RAGEvaluator:
    """
    Evaluate RAG performance on legal Q&A
    """

    def __init__(self):
        self.faithfulness_evaluator = load_evaluator("qa")

    def evaluate_batch(
        self,
        questions: list[str],
        ground_truth: list[str],
        generated_answers: list[str],
        contexts: list[list[str]]
    ) -> dict:
        """
        RAGAS evaluation:
        - Faithfulness: Answer grounded in context
        - Answer Relevancy: Addresses the question
        - Context Precision: Retrieved docs are relevant
        - Context Recall: All needed info retrieved
        """

        dataset = {
            "question": questions,
            "answer": generated_answers,
            "contexts": contexts,
            "ground_truth": ground_truth
        }

        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        return results

    def check_citation_coverage(
        self,
        answer: str,
        citations: list[dict]
    ) -> float:
        """
        Verify every claim has a citation
        """
        pass
```

### 8.2 Prometheus Metrics

```python
# services/api-gateway/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
query_requests_total = Counter(
    'conformai_query_requests_total',
    'Total compliance queries',
    ['endpoint', 'status']
)

query_duration_seconds = Histogram(
    'conformai_query_duration_seconds',
    'Query processing time',
    ['endpoint']
)

# RAG metrics
retrieval_documents_count = Histogram(
    'conformai_retrieval_documents',
    'Number of documents retrieved',
    buckets=[0, 5, 10, 20, 50]
)

confidence_score = Histogram(
    'conformai_confidence_score',
    'Answer confidence scores',
    buckets=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
)

# Data freshness
regulation_last_update = Gauge(
    'conformai_regulation_last_update_timestamp',
    'Last update timestamp for regulations',
    ['regulation']
)
```

---

## Phase 9: Security & Compliance

### 9.1 API Authentication

```python
# services/api-gateway/auth/api_key.py

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import hashlib

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key against database
    """
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Check in database
    valid = await check_api_key_in_db(key_hash)

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    # Log usage
    await log_api_usage(key_hash)

    return api_key
```

### 9.2 Data Privacy

```python
# services/api-gateway/privacy/pii_filter.py

class PIIFilter:
    """
    Detect and redact PII from queries before logging
    """

    def filter_query(self, query: str) -> str:
        """
        Remove:
        - Names
        - Email addresses
        - Phone numbers
        - Company identifiers
        """
        pass
```

---

## Implementation Timeline

### Week 1-2: Foundation
- ✅ Project setup & dependency installation
- ✅ Docker Compose configuration
- ✅ PostgreSQL + Qdrant setup
- ✅ Basic Airflow DAGs

### Week 3-4: Data Pipeline
- ✅ EUR-Lex API integration
- ✅ Legal document parser
- ✅ Chunking & embedding pipeline
- ✅ Vector DB indexing

### Week 5-6: Retrieval Layer
- ✅ LangChain retrieval setup
- ✅ Hybrid search implementation
- ✅ Query classification
- ✅ Reranking

### Week 7-8: RAG Orchestration
- ✅ LangGraph state machine
- ✅ Prompt engineering
- ✅ Citation extraction
- ✅ Grounding validation

### Week 9-10: API & Testing
- ✅ FastAPI endpoints
- ✅ Authentication
- ✅ Integration tests
- ✅ RAG evaluation

### Week 11-12: Deployment & Polish
- ✅ Kubernetes manifests
- ✅ Monitoring setup
- ✅ Documentation
- ✅ Demo UI

---

## Next Steps

1. **Initialize Project Structure**
   ```bash
   mkdir -p services/{api-gateway,rag-service,retrieval-service,data-pipeline}
   mkdir -p airflow/{dags,plugins,config}
   mkdir -p infrastructure/{docker,k8s}
   ```

2. **Set Up Python Environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Configure Environment Variables**
   ```bash
   cp .env.example .env
   # Add API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
   ```

4. **Start Infrastructure**
   ```bash
   docker-compose up -d postgres qdrant redis
   ```

5. **Initialize Databases**
   ```bash
   alembic upgrade head
   python scripts/init_vectordb.py
   ```

6. **Run First Data Pipeline**
   ```bash
   airflow dags trigger eurlex_daily_ingestion
   ```

---

## Success Metrics

- **Retrieval Quality**: >0.8 context precision
- **Faithfulness**: >0.9 (answers grounded in sources)
- **Citation Coverage**: 100% of claims cited
- **Latency**: <2s per query (p95)
- **Data Freshness**: <24h lag from official sources
- **Uptime**: 99.9% availability

---

**Ready to build a production-grade legal RAG system!** 🚀
