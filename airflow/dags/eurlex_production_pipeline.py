"""
EUR-Lex Production Pipeline DAG

Complete pipeline for:
1. Fetching new/updated EU regulations
2. Parsing documents
3. Chunking with legal-aware strategy
4. Generating embeddings
5. Indexing into Qdrant

Runs daily at 2 AM.
"""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    "owner": "conformai",
    "depends_on_past": False,
    "email": ["alerts@conformai.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}


def check_for_new_documents(**context):
    """
    Task 1: Check EUR-Lex for new/updated AI-related documents.

    For simplicity, we'll check the fallback CELEX list.
    In production, you'd query EUR-Lex SPARQL endpoint.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')

    from shared.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Checking for new AI-related documents...")

    # Fallback CELEX IDs (known regulations)
    # In production, fetch from EUR-Lex API
    celex_ids = [
        "32016R0679",  # GDPR
        "52021PC0206",  # AI Act
        "52020PC0767",  # Data Governance Act
    ]

    logger.info(f"Found {len(celex_ids)} documents to check: {celex_ids}")

    # Push to XCom for next task
    context["ti"].xcom_push(key="celex_ids", value=celex_ids)

    return len(celex_ids)


def download_documents(**context):
    """
    Task 2: Download documents from EUR-Lex (if not already downloaded).
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/data-pipeline/src')

    from clients.eurlex_client import EURLexClient
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)

    # Get CELEX IDs from previous task
    ti = context["ti"]
    celex_ids = ti.xcom_pull(key="celex_ids", task_ids="check_for_new_documents")

    if not celex_ids:
        logger.info("No documents to download")
        return 0

    # Setup paths
    raw_dir = Path("/opt/airflow/data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Initialize EUR-Lex client
    client = EURLexClient(
        timeout=120,
        max_retries=3
    )

    downloaded = 0

    for celex in celex_ids:
        output_path = raw_dir / f"{celex}.xml"

        # Skip if already exists
        if output_path.exists():
            logger.info(f"Document {celex} already exists, skipping download")
            continue

        try:
            logger.info(f"Downloading {celex}...")
            client.download_document_to_file(
                celex=celex,
                output_path=output_path,
                format="xml",
                language="EN",
            )
            downloaded += 1
            logger.info(f"✓ Downloaded {celex}")

        except Exception as e:
            logger.error(f"Failed to download {celex}: {e}")
            continue

    client.close()

    logger.info(f"Downloaded {downloaded} new documents")

    # Push downloaded list for next task
    ti.xcom_push(key="celex_ids_downloaded", value=celex_ids)

    return downloaded


def parse_documents(**context):
    """
    Task 3: Parse downloaded XML documents into structured format.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/data-pipeline/src')

    from parsers.legal_parser import LegalDocumentParser
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)

    # Get document list from previous task
    ti = context["ti"]
    celex_ids = ti.xcom_pull(key="celex_ids_downloaded", task_ids="download_documents")

    if not celex_ids:
        logger.info("No documents to parse")
        return 0

    # Setup paths
    raw_dir = Path("/opt/airflow/data/raw")
    processed_dir = Path("/opt/airflow/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Initialize parser
    parser = LegalDocumentParser()

    parsed_count = 0

    for celex in celex_ids:
        doc_path = raw_dir / f"{celex}.xml"

        if not doc_path.exists():
            logger.warning(f"Document {celex} not found, skipping")
            continue

        try:
            logger.info(f"Parsing {celex}...")

            # Parse document
            document = parser.parse(doc_path, regulation=None)

            # Save parsed document
            parsed_path = processed_dir / f"{celex}.pkl"
            with parsed_path.open("wb") as f:
                pickle.dump(document, f)

            parsed_count += 1
            logger.info(f"✓ Parsed {celex}")

        except Exception as e:
            logger.error(f"Failed to parse {celex}: {e}")
            continue

    logger.info(f"Parsed {parsed_count} documents")

    # Push for next task
    ti.xcom_push(key="celex_ids_parsed", value=celex_ids)

    return parsed_count


def chunk_documents(**context):
    """
    Task 4: Chunk documents using legal-aware strategy.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/data-pipeline/src')

    from chunking.legal_chunker import LegalChunker
    from shared.config.settings import get_settings
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # Get document list
    ti = context["ti"]
    celex_ids = ti.xcom_pull(key="celex_ids_parsed", task_ids="parse_documents")

    if not celex_ids:
        logger.info("No documents to chunk")
        return 0

    # Setup paths
    processed_dir = Path("/opt/airflow/data/processed")
    chunks_dir = processed_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Initialize chunker
    chunker = LegalChunker(
        max_chunk_tokens=settings.chunk_size,
        overlap_sentences=2,
        tokenizer_type="tiktoken",
        encoding_name="cl100k_base",
    )

    total_chunks = 0

    for celex in celex_ids:
        parsed_path = processed_dir / f"{celex}.pkl"

        if not parsed_path.exists():
            logger.warning(f"Parsed document {celex} not found, skipping")
            continue

        try:
            logger.info(f"Chunking {celex}...")

            # Load parsed document
            with parsed_path.open("rb") as f:
                document = pickle.load(f)

            # Chunk document (automatically filters bad chunks now)
            chunks = chunker.chunk_document(document)

            # Save chunks
            chunks_path = chunks_dir / f"{celex}_chunks.pkl"
            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)

            total_chunks += len(chunks)
            logger.info(f"✓ Created {len(chunks)} chunks for {celex}")

        except Exception as e:
            logger.error(f"Failed to chunk {celex}: {e}")
            continue

    logger.info(f"Created {total_chunks} total chunks")

    # Push for next task
    ti.xcom_push(key="celex_ids_chunked", value=celex_ids)
    ti.xcom_push(key="total_chunks", value=total_chunks)

    return total_chunks


def generate_embeddings(**context):
    """
    Task 5: Generate embeddings for all chunks.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/data-pipeline/src')

    from embeddings.embedding_generator import EmbeddingGenerator
    from shared.config.settings import get_settings
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # Get document list
    ti = context["ti"]
    celex_ids = ti.xcom_pull(key="celex_ids_chunked", task_ids="chunk_documents")

    if not celex_ids:
        logger.info("No documents to embed")
        return 0

    # Setup paths
    chunks_dir = Path("/opt/airflow/data/processed/chunks")
    embeddings_dir = Path("/opt/airflow/data/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Initialize embedder
    embedder = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=50,  # Safer batch size
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    total_embedded = 0

    for celex in celex_ids:
        chunks_path = chunks_dir / f"{celex}_chunks.pkl"
        embeddings_path = embeddings_dir / f"{celex}_embedded.pkl"

        # Skip if embeddings already exist and are valid
        if embeddings_path.exists():
            try:
                with embeddings_path.open("rb") as f:
                    existing = pickle.load(f)
                if existing and len(existing) > 0:
                    logger.info(f"Embeddings for {celex} already exist ({len(existing)} chunks), skipping")
                    total_embedded += len(existing)
                    continue
            except Exception:
                logger.warning(f"Invalid embeddings for {celex}, regenerating...")

        if not chunks_path.exists():
            logger.warning(f"Chunks for {celex} not found, skipping")
            continue

        try:
            logger.info(f"Generating embeddings for {celex}...")

            # Load chunks
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)

            if not chunks:
                logger.warning(f"No chunks found for {celex}")
                continue

            # Generate embeddings
            embedded_chunks = embedder.generate_embeddings(chunks, normalize=True)

            # Save embeddings
            with embeddings_path.open("wb") as f:
                pickle.dump(embedded_chunks, f)

            total_embedded += len(embedded_chunks)
            logger.info(f"✓ Generated {len(embedded_chunks)} embeddings for {celex}")

        except Exception as e:
            logger.error(f"Failed to generate embeddings for {celex}: {e}")
            continue

    logger.info(f"Generated {total_embedded} total embeddings")

    # Push for next task
    ti.xcom_push(key="celex_ids_embedded", value=celex_ids)
    ti.xcom_push(key="total_embedded", value=total_embedded)

    return total_embedded


def index_to_qdrant(**context):
    """
    Task 6: Index embeddings into Qdrant vector database.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/data-pipeline/src')

    from indexing.qdrant_indexer import QdrantIndexer
    from shared.config.settings import get_settings
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # Get document list
    ti = context["ti"]
    celex_ids = ti.xcom_pull(key="celex_ids_embedded", task_ids="generate_embeddings")

    if not celex_ids:
        logger.info("No documents to index")
        return 0

    # Setup paths
    embeddings_dir = Path("/opt/airflow/data/embeddings")

    # Initialize indexer
    indexer = QdrantIndexer(
        collection_name=settings.qdrant_collection_name,
    )

    # Ensure collection exists (but don't recreate)
    try:
        indexer.create_collection(recreate=False)
    except Exception:
        # Collection probably already exists
        pass

    total_indexed = 0

    for celex in celex_ids:
        embeddings_path = embeddings_dir / f"{celex}_embedded.pkl"

        if not embeddings_path.exists():
            logger.warning(f"Embeddings for {celex} not found, skipping")
            continue

        try:
            logger.info(f"Indexing {celex}...")

            # Load embeddings
            with embeddings_path.open("rb") as f:
                embedded_chunks = pickle.load(f)

            if not embedded_chunks:
                logger.warning(f"No embedded chunks for {celex}")
                continue

            # Index chunks
            indexed = indexer.index_chunks(embedded_chunks, batch_size=100, show_progress=False)

            total_indexed += indexed
            logger.info(f"✓ Indexed {indexed} chunks for {celex}")

        except Exception as e:
            logger.error(f"Failed to index {celex}: {e}")
            continue

    indexer.close()

    logger.info(f"Indexed {total_indexed} total chunks")

    return total_indexed


def validate_pipeline(**context):
    """
    Task 7: Validate that pipeline completed successfully.
    """
    import sys
    sys.path.insert(0, '/opt/airflow')
    sys.path.insert(0, '/opt/airflow/services/retrieval-service/src')

    from retrieval.qdrant_client import get_qdrant_store
    from shared.config.settings import get_settings
    from shared.utils.logger import get_logger

    logger = get_logger(__name__)
    settings = get_settings()

    # Get stats from previous tasks
    ti = context["ti"]
    total_chunks = ti.xcom_pull(key="total_chunks", task_ids="chunk_documents") or 0
    total_embedded = ti.xcom_pull(key="total_embedded", task_ids="generate_embeddings") or 0
    total_indexed = ti.xcom_pull(task_ids="index_to_qdrant") or 0

    logger.info("=== Pipeline Summary ===")
    logger.info(f"Total chunks created: {total_chunks}")
    logger.info(f"Total embeddings generated: {total_embedded}")
    logger.info(f"Total chunks indexed: {total_indexed}")

    # Verify Qdrant
    try:
        import asyncio

        store = get_qdrant_store()

        async def check():
            info = await store.get_collection_info()
            return info

        info = asyncio.run(check())

        logger.info(f"Qdrant collection: {info.get('name')}")
        logger.info(f"Total points in Qdrant: {info.get('points_count')}")

        # Success criteria
        if info.get('points_count', 0) > 0:
            logger.info("✓ Pipeline validation PASSED")
            return True
        else:
            logger.error("✗ Pipeline validation FAILED: No points in Qdrant")
            raise ValueError("Qdrant has no documents")

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


# Define DAG
with DAG(
    "eurlex_production_pipeline",
    default_args=default_args,
    description="Production pipeline for EU regulatory compliance data",
    schedule_interval="0 2 * * *",  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=["production", "data-pipeline", "eurlex"],
) as dag:

    # Task 1: Check for new documents
    task_check = PythonOperator(
        task_id="check_for_new_documents",
        python_callable=check_for_new_documents,
    )

    # Task 2: Download documents
    task_download = PythonOperator(
        task_id="download_documents",
        python_callable=download_documents,
    )

    # Task 3: Parse documents
    task_parse = PythonOperator(
        task_id="parse_documents",
        python_callable=parse_documents,
    )

    # Task 4: Chunk documents
    task_chunk = PythonOperator(
        task_id="chunk_documents",
        python_callable=chunk_documents,
    )

    # Task 5: Generate embeddings
    task_embed = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
    )

    # Task 6: Index to Qdrant
    task_index = PythonOperator(
        task_id="index_to_qdrant",
        python_callable=index_to_qdrant,
    )

    # Task 7: Validate pipeline
    task_validate = PythonOperator(
        task_id="validate_pipeline",
        python_callable=validate_pipeline,
    )

    # Define task dependencies
    task_check >> task_download >> task_parse >> task_chunk >> task_embed >> task_index >> task_validate
