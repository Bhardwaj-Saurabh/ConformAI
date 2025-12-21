"""
EUR-Lex Daily Ingestion DAG

Scheduled pipeline to check for new/updated EU regulations daily.
Uses the services/data-pipeline library components.
"""

# Import data-pipeline components
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator

from airflow import DAG

sys.path.insert(0, '/opt/airflow')

from services.data_pipeline.src.chunking import LegalChunker
from services.data_pipeline.src.clients import EURLexClient
from services.data_pipeline.src.embeddings import EmbeddingGenerator
from services.data_pipeline.src.indexing import QdrantIndexer
from services.data_pipeline.src.parsers import LegalDocumentParser


def fetch_new_documents(**context):
    """Task 1: Fetch new/updated documents from EUR-Lex."""
    client = EURLexClient()

    # Search for AI-related documents updated in last 24 hours
    docs = client.search_ai_related_documents(
        start_date=datetime.now() - timedelta(days=1),
        limit=10
    )

    # Download each document
    celex_ids = [doc['celex'] for doc in docs if 'celex' in doc]

    # Push to XCom for next task
    context['task_instance'].xcom_push(key='celex_ids', value=celex_ids)

    return f"Found {len(celex_ids)} documents"


def parse_documents(**context):
    """Task 2: Parse downloaded documents."""
    celex_ids = context['task_instance'].xcom_pull(key='celex_ids')

    if not celex_ids:
        return "No documents to parse"

    parser = LegalDocumentParser()
    parsed_count = 0

    for celex in celex_ids:
        doc_path = Path(f"/opt/airflow/data/raw/{celex}.xml")
        if doc_path.exists():
            document = parser.parse(doc_path)
            parsed_count += 1

    context['task_instance'].xcom_push(key='parsed_count', value=parsed_count)
    return f"Parsed {parsed_count} documents"


def chunk_and_embed(**context):
    """Task 3: Chunk and generate embeddings."""
    celex_ids = context['task_instance'].xcom_pull(key='celex_ids')

    if not celex_ids:
        return "No documents to chunk"

    chunker = LegalChunker(max_chunk_tokens=512)
    embedder = EmbeddingGenerator(batch_size=50)

    total_chunks = 0

    for celex in celex_ids:
        # Load parsed document
        # Chunk it
        # Generate embeddings
        # Save for indexing
        pass  # Implementation here

    return f"Created {total_chunks} embedded chunks"


def index_to_qdrant(**context):
    """Task 4: Index embeddings to Qdrant."""
    # Index all new embeddings
    indexer = QdrantIndexer()

    # Index chunks
    indexed = indexer.index_chunks(chunks)

    return f"Indexed {indexed} chunks"


# DAG Definition
default_args = {
    'owner': 'conformai',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'eurlex_daily_ingestion',
    default_args=default_args,
    description='Daily ingestion of new EUR-Lex documents',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['eurlex', 'data-pipeline', 'production'],
) as dag:

    # Task dependencies
    fetch = PythonOperator(
        task_id='fetch_documents',
        python_callable=fetch_new_documents,
    )

    parse = PythonOperator(
        task_id='parse_documents',
        python_callable=parse_documents,
    )

    chunk_embed = PythonOperator(
        task_id='chunk_and_embed',
        python_callable=chunk_and_embed,
    )

    index = PythonOperator(
        task_id='index_to_qdrant',
        python_callable=index_to_qdrant,
    )

    # Pipeline flow
    fetch >> parse >> chunk_embed >> index
