"""
Airflow DAG for EU Legal Documents Data Pipeline

Orchestrates the complete data pipeline:
1. Document discovery and download from EUR-Lex
2. Document parsing and extraction
3. Legal-aware chunking
4. Embedding generation
5. Vector database indexing

Schedule: Daily at 2 AM UTC
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "services" / "data-pipeline" / "src"))

from shared.utils.logger import get_logger

logger = get_logger(__name__)

# DAG default arguments
default_args = {
    "owner": "conformai",
    "depends_on_past": False,
    "email": os.getenv("ALERT_EMAIL", "admin@conformai.com"),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}

# DAG configuration
dag = DAG(
    dag_id="eu_legal_documents_pipeline",
    default_args=default_args,
    description="Daily EU legal documents ingestion and indexing pipeline",
    schedule_interval="0 2 * * *",  # Daily at 2 AM UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["data-pipeline", "legal-documents", "eu-regulations"],
    max_active_runs=1,
)


def check_prerequisites(**context):
    """
    Check that all required services and configurations are available.

    Validates:
    - API keys (Anthropic, OpenAI)
    - Database connections (Qdrant, PostgreSQL)
    - Required directories
    - Minimum disk space
    """
    import shutil

    from qdrant_client import QdrantClient

    from shared.config import get_settings

    logger.info("Checking pipeline prerequisites...")

    settings = get_settings()
    checks_passed = []
    checks_failed = []

    # Check API keys
    if settings.anthropic_api_key and len(settings.anthropic_api_key) > 10:
        checks_passed.append("✓ Anthropic API key configured")
    else:
        checks_failed.append("✗ Anthropic API key missing or invalid")

    if settings.openai_api_key and len(settings.openai_api_key) > 10:
        checks_passed.append("✓ OpenAI API key configured")
    else:
        checks_failed.append("✗ OpenAI API key missing or invalid")

    # Check Qdrant connection
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collections = client.get_collections()
        checks_passed.append(f"✓ Qdrant connected ({len(collections.collections)} collections)")
        client.close()
    except Exception as e:
        checks_failed.append(f"✗ Qdrant connection failed: {str(e)}")

    # Check disk space (require at least 10GB free)
    try:
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024**3)
        if free_gb >= 10:
            checks_passed.append(f"✓ Disk space: {free_gb:.1f} GB free")
        else:
            checks_failed.append(f"✗ Low disk space: {free_gb:.1f} GB free (minimum 10 GB required)")
    except Exception as e:
        checks_failed.append(f"✗ Could not check disk space: {str(e)}")

    # Check required directories
    required_dirs = ["data/raw", "data/processed", "data/embeddings"]
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            checks_passed.append(f"✓ Directory exists: {dir_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            checks_passed.append(f"✓ Created directory: {dir_path}")

    # Log results
    logger.info("Prerequisite checks:")
    for check in checks_passed:
        logger.info(check)
    for check in checks_failed:
        logger.error(check)

    # Push to XCom for downstream tasks
    context["task_instance"].xcom_push(key="checks_passed", value=len(checks_passed))
    context["task_instance"].xcom_push(key="checks_failed", value=len(checks_failed))

    if checks_failed:
        raise ValueError(f"Prerequisite checks failed: {len(checks_failed)} failures")

    logger.info(f"All prerequisite checks passed ({len(checks_passed)} checks)")


def discover_documents(**context):
    """
    Discover new documents from EUR-Lex.

    Returns:
        List of CELEX IDs to process
    """
    from datetime import date, timedelta

    from clients import EURLexClient

    from shared.config import get_settings

    logger.info("Discovering EU legal documents from EUR-Lex...")

    settings = get_settings()

    # Look for documents from the last 30 days
    start_date = date.today() - timedelta(days=30)

    client = EURLexClient(
        sparql_endpoint=settings.eurlex_api_base_url,
        timeout=120,
        max_retries=5,
    )

    try:
        # Search for AI-related documents
        docs = client.search_ai_related_documents(start_date=start_date, limit=20)
        celex_ids = [doc["celex"] for doc in docs if doc.get("celex")]

        # Remove duplicates
        celex_ids = list(set(celex_ids))

        logger.info(f"Found {len(celex_ids)} documents to process")

        # Push to XCom
        context["task_instance"].xcom_push(key="celex_ids", value=celex_ids)
        context["task_instance"].xcom_push(key="documents_found", value=len(celex_ids))

        return celex_ids

    finally:
        client.close()


def download_documents(**context):
    """Download documents from EUR-Lex."""
    from clients import EURLexClient

    from shared.config import get_settings

    # Get CELEX IDs from previous task
    celex_ids = context["task_instance"].xcom_pull(
        task_ids="discover_documents", key="celex_ids"
    )

    if not celex_ids:
        logger.warning("No documents to download")
        return

    logger.info(f"Downloading {len(celex_ids)} documents...")

    settings = get_settings()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = EURLexClient(
        sparql_endpoint=settings.eurlex_api_base_url,
        timeout=120,
        max_retries=5,
    )

    downloaded = []
    failed = []

    try:
        for idx, celex in enumerate(celex_ids, 1):
            output_path = raw_dir / f"{celex}.xml"

            # Skip if already exists
            if output_path.exists():
                logger.info(f"Skipping {celex} (already downloaded)")
                downloaded.append(str(output_path))
                continue

            try:
                logger.info(f"Downloading {idx}/{len(celex_ids)}: {celex}")
                client.download_document_to_file(
                    celex=celex,
                    output_path=output_path,
                    format="xml",
                    language="EN",
                )
                downloaded.append(str(output_path))
                logger.info(f"✓ Downloaded {celex}")
            except Exception as e:
                logger.error(f"Failed to download {celex}: {str(e)}")
                failed.append(celex)

        logger.info(f"Downloaded {len(downloaded)} documents, {len(failed)} failed")

        # Push to XCom
        context["task_instance"].xcom_push(key="downloaded_paths", value=downloaded)
        context["task_instance"].xcom_push(key="download_errors", value=len(failed))

    finally:
        client.close()


def parse_documents(**context):
    """Parse downloaded legal documents."""
    import pickle

    from clients import EURLexClient
    from parsers import LegalDocumentParser

    from shared.config import get_settings

    # Get downloaded paths from previous task
    downloaded_paths = context["task_instance"].xcom_pull(
        task_ids="download_documents", key="downloaded_paths"
    )

    if not downloaded_paths:
        logger.warning("No documents to parse")
        return

    logger.info(f"Parsing {len(downloaded_paths)} documents...")

    settings = get_settings()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    parser = LegalDocumentParser()
    eurlex_client = EURLexClient(
        sparql_endpoint=settings.eurlex_api_base_url,
        timeout=120,
        max_retries=5,
    )

    parsed = []
    failed = []
    total_chapters = 0
    total_articles = 0

    try:
        for idx, doc_path_str in enumerate(downloaded_paths, 1):
            doc_path = Path(doc_path_str)
            celex = doc_path.stem
            parsed_path = processed_dir / f"{celex}.pkl"

            # Skip if already parsed
            if parsed_path.exists():
                logger.info(f"Skipping {celex} (already parsed)")
                parsed.append(str(parsed_path))
                continue

            try:
                logger.info(f"Parsing {idx}/{len(downloaded_paths)}: {celex}")

                # Get metadata
                try:
                    regulation = eurlex_client.extract_celex_metadata(celex)
                except Exception as e:
                    logger.warning(f"Metadata lookup failed for {celex}: {str(e)}")
                    regulation = None

                # Parse document
                document = parser.parse(doc_path, regulation=regulation)

                # Save parsed document
                with parsed_path.open("wb") as f:
                    pickle.dump(document, f)

                chapters = len(document.chapters)
                articles = sum(len(ch.articles) for ch in document.chapters)
                total_chapters += chapters
                total_articles += articles

                parsed.append(str(parsed_path))
                logger.info(f"✓ Parsed {celex}: {chapters} chapters, {articles} articles")

            except Exception as e:
                logger.error(f"Failed to parse {celex}: {str(e)}")
                failed.append(celex)

        logger.info(
            f"Parsed {len(parsed)} documents ({total_chapters} chapters, {total_articles} articles), {len(failed)} failed"
        )

        # Push to XCom
        context["task_instance"].xcom_push(key="parsed_paths", value=parsed)
        context["task_instance"].xcom_push(key="parse_errors", value=len(failed))
        context["task_instance"].xcom_push(key="total_chapters", value=total_chapters)
        context["task_instance"].xcom_push(key="total_articles", value=total_articles)

    finally:
        eurlex_client.close()


def chunk_documents(**context):
    """Chunk parsed documents."""
    import pickle

    from chunking import LegalChunker

    from shared.config import get_settings

    # Get parsed paths from previous task
    parsed_paths = context["task_instance"].xcom_pull(
        task_ids="parse_documents", key="parsed_paths"
    )

    if not parsed_paths:
        logger.warning("No documents to chunk")
        return

    logger.info(f"Chunking {len(parsed_paths)} documents...")

    settings = get_settings()
    chunks_dir = PROJECT_ROOT / "data" / "processed" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunker = LegalChunker(
        max_chunk_tokens=settings.chunk_size,
        overlap_sentences=2,
        tokenizer_type="tiktoken",
        encoding_name="cl100k_base",
    )

    chunked = []
    failed = []
    total_chunks = 0

    for idx, parsed_path_str in enumerate(parsed_paths, 1):
        parsed_path = Path(parsed_path_str)
        celex = parsed_path.stem
        chunks_path = chunks_dir / f"{celex}_chunks.pkl"

        # Skip if already chunked
        if chunks_path.exists():
            logger.info(f"Skipping {celex} (already chunked)")
            chunked.append(str(chunks_path))
            continue

        try:
            logger.info(f"Chunking {idx}/{len(parsed_paths)}: {celex}")

            # Load parsed document
            with parsed_path.open("rb") as f:
                document = pickle.load(f)

            # Chunk document
            chunks = chunker.chunk_document(document)

            # Filter out invalid chunks
            chunks = [
                chunk for chunk in chunks
                if hasattr(chunk, 'text') and chunk.text and len(chunk.text.strip()) >= 10
            ]

            if not chunks:
                logger.warning(f"No valid chunks for {celex}")
                failed.append(celex)
                continue

            # Save chunks
            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)

            total_chunks += len(chunks)
            chunked.append(str(chunks_path))
            logger.info(f"✓ Chunked {celex}: {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to chunk {celex}: {str(e)}")
            failed.append(celex)

    logger.info(f"Chunked {len(chunked)} documents ({total_chunks} chunks), {len(failed)} failed")

    # Push to XCom
    context["task_instance"].xcom_push(key="chunked_paths", value=chunked)
    context["task_instance"].xcom_push(key="chunk_errors", value=len(failed))
    context["task_instance"].xcom_push(key="total_chunks", value=total_chunks)


def generate_embeddings(**context):
    """Generate embeddings for chunks."""
    import pickle

    from embeddings import EmbeddingGenerator

    from shared.config import get_settings

    # Get chunked paths from previous task
    chunked_paths = context["task_instance"].xcom_pull(
        task_ids="chunk_documents", key="chunked_paths"
    )

    if not chunked_paths:
        logger.warning("No documents to embed")
        return

    logger.info(f"Generating embeddings for {len(chunked_paths)} documents...")

    settings = get_settings()
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    generator = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=50,
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    embedded = []
    failed = []
    total_embeddings = 0

    for idx, chunks_path_str in enumerate(chunked_paths, 1):
        chunks_path = Path(chunks_path_str)
        celex = chunks_path.stem.replace("_chunks", "")
        embedded_path = embeddings_dir / f"{celex}_embedded.pkl"

        # Skip if already embedded
        if embedded_path.exists():
            logger.info(f"Skipping {celex} (already embedded)")
            embedded.append(str(embedded_path))
            continue

        try:
            logger.info(f"Embedding {idx}/{len(chunked_paths)}: {celex}")

            # Load chunks
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)

            # Generate embeddings
            embedded_chunks = generator.generate_embeddings(chunks, normalize=True)

            # Save embedded chunks
            with embedded_path.open("wb") as f:
                pickle.dump(embedded_chunks, f)

            total_embeddings += len(embedded_chunks)
            embedded.append(str(embedded_path))
            logger.info(f"✓ Embedded {celex}: {len(embedded_chunks)} embeddings")

        except Exception as e:
            logger.error(f"Failed to embed {celex}: {str(e)}")
            failed.append(celex)

    logger.info(f"Embedded {len(embedded)} documents ({total_embeddings} embeddings), {len(failed)} failed")

    # Push to XCom
    context["task_instance"].xcom_push(key="embedded_paths", value=embedded)
    context["task_instance"].xcom_push(key="embed_errors", value=len(failed))
    context["task_instance"].xcom_push(key="total_embeddings", value=total_embeddings)


def index_to_qdrant(**context):
    """Index embeddings to Qdrant vector database."""
    import pickle

    from indexing import QdrantIndexer

    # Get embedded paths from previous task
    embedded_paths = context["task_instance"].xcom_pull(
        task_ids="generate_embeddings", key="embedded_paths"
    )

    if not embedded_paths:
        logger.warning("No embeddings to index")
        return

    logger.info(f"Indexing {len(embedded_paths)} documents to Qdrant...")

    indexer = QdrantIndexer()

    # Ensure collection exists (don't recreate, just ensure it exists)
    indexer.create_collection(recreate=False)

    total_indexed = 0
    failed = []

    try:
        for idx, embedded_path_str in enumerate(embedded_paths, 1):
            embedded_path = Path(embedded_path_str)
            celex = embedded_path.stem.replace("_embedded", "")

            try:
                logger.info(f"Indexing {idx}/{len(embedded_paths)}: {celex}")

                # Load embedded chunks
                with embedded_path.open("rb") as f:
                    embedded_chunks = pickle.load(f)

                # Index to Qdrant
                indexed = indexer.index_chunks(embedded_chunks, batch_size=100, show_progress=False)
                total_indexed += indexed
                logger.info(f"✓ Indexed {celex}: {indexed} points")

            except Exception as e:
                logger.error(f"Failed to index {celex}: {str(e)}")
                failed.append(celex)

        logger.info(f"Indexed {total_indexed} points, {len(failed)} documents failed")

        # Push to XCom
        context["task_instance"].xcom_push(key="total_indexed", value=total_indexed)
        context["task_instance"].xcom_push(key="index_errors", value=len(failed))

    finally:
        indexer.close()


def send_completion_notification(**context):
    """Send pipeline completion notification with summary."""
    from airflow.utils.email import send_email

    # Gather metrics from XCom
    documents_found = context["task_instance"].xcom_pull(
        task_ids="discover_documents", key="documents_found"
    ) or 0

    download_errors = context["task_instance"].xcom_pull(
        task_ids="download_documents", key="download_errors"
    ) or 0

    parse_errors = context["task_instance"].xcom_pull(
        task_ids="parse_documents", key="parse_errors"
    ) or 0

    chunk_errors = context["task_instance"].xcom_pull(
        task_ids="chunk_documents", key="chunk_errors"
    ) or 0

    embed_errors = context["task_instance"].xcom_pull(
        task_ids="generate_embeddings", key="embed_errors"
    ) or 0

    index_errors = context["task_instance"].xcom_pull(
        task_ids="index_to_qdrant", key="index_errors"
    ) or 0

    total_indexed = context["task_instance"].xcom_pull(
        task_ids="index_to_qdrant", key="total_indexed"
    ) or 0

    total_chapters = context["task_instance"].xcom_pull(
        task_ids="parse_documents", key="total_chapters"
    ) or 0

    total_articles = context["task_instance"].xcom_pull(
        task_ids="parse_documents", key="total_articles"
    ) or 0

    total_embeddings = context["task_instance"].xcom_pull(
        task_ids="generate_embeddings", key="total_embeddings"
    ) or 0

    # Calculate success rate
    total_errors = download_errors + parse_errors + chunk_errors + embed_errors + index_errors
    success_rate = ((documents_found - total_errors) / documents_found * 100) if documents_found > 0 else 0

    # Create summary
    summary = f"""
    EU Legal Documents Pipeline - Execution Summary
    ================================================

    Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Documents Processed:
    - Found: {documents_found}
    - Successfully Indexed: {documents_found - total_errors}
    - Success Rate: {success_rate:.1f}%

    Pipeline Stages:
    - Download Errors: {download_errors}
    - Parse Errors: {parse_errors}
    - Chunk Errors: {chunk_errors}
    - Embed Errors: {embed_errors}
    - Index Errors: {index_errors}

    Metrics:
    - Total Chapters: {total_chapters}
    - Total Articles: {total_articles}
    - Total Embeddings: {total_embeddings}
    - Total Points Indexed: {total_indexed}

    Status: {"✓ SUCCESS" if total_errors == 0 else f"⚠ COMPLETED WITH {total_errors} ERRORS"}
    """

    logger.info(summary)

    # Push summary to XCom
    context["task_instance"].xcom_push(key="pipeline_summary", value=summary)

    # Send email if configured
    alert_email = os.getenv("ALERT_EMAIL")
    if alert_email:
        try:
            send_email(
                to=[alert_email],
                subject=f"ConformAI Pipeline {'Success' if total_errors == 0 else 'Completed with Errors'}",
                html_content=f"<pre>{summary}</pre>",
            )
            logger.info(f"Sent notification email to {alert_email}")
        except Exception as e:
            logger.warning(f"Failed to send email: {str(e)}")


# Define tasks
check_prerequisites_task = PythonOperator(
    task_id="check_prerequisites",
    python_callable=check_prerequisites,
    dag=dag,
)

discover_documents_task = PythonOperator(
    task_id="discover_documents",
    python_callable=discover_documents,
    dag=dag,
)

download_documents_task = PythonOperator(
    task_id="download_documents",
    python_callable=download_documents,
    dag=dag,
)

parse_documents_task = PythonOperator(
    task_id="parse_documents",
    python_callable=parse_documents,
    dag=dag,
)

chunk_documents_task = PythonOperator(
    task_id="chunk_documents",
    python_callable=chunk_documents,
    dag=dag,
)

generate_embeddings_task = PythonOperator(
    task_id="generate_embeddings",
    python_callable=generate_embeddings,
    dag=dag,
)

index_to_qdrant_task = PythonOperator(
    task_id="index_to_qdrant",
    python_callable=index_to_qdrant,
    dag=dag,
)

send_notification_task = PythonOperator(
    task_id="send_completion_notification",
    python_callable=send_completion_notification,
    trigger_rule=TriggerRule.ALL_DONE,  # Run even if upstream tasks fail
    dag=dag,
)

# Define task dependencies
(
    check_prerequisites_task
    >> discover_documents_task
    >> download_documents_task
    >> parse_documents_task
    >> chunk_documents_task
    >> generate_embeddings_task
    >> index_to_qdrant_task
    >> send_notification_task
)
