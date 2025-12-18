"""
Document Processing DAG

Complete pipeline for processing legal documents:
1. Parse documents (XML/HTML/PDF → structured data)
2. Chunk with legal-aware strategy
3. Generate embeddings
4. Index to Qdrant vector database

Triggered after EUR-Lex ingestion DAG completes.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add services to path
sys.path.insert(0, "/opt/airflow")

# Default arguments
default_args = {
    "owner": "conformai",
    "depends_on_past": False,
    "email": ["admin@conformai.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(hours=4),
}


def parse_documents(**context):
    """
    Parse downloaded legal documents.

    Converts XML/HTML/PDF to structured LegalDocument objects.
    """
    from services.data_pipeline.src.clients import EURLexClient
    from services.data_pipeline.src.parsers import LegalDocumentParser
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Starting document parsing...")

    # Get paths
    raw_dir = Path("/opt/airflow/data/raw")
    processed_dir = Path("/opt/airflow/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find XML files
    xml_files = list(raw_dir.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML files to parse")

    if not xml_files:
        logger.warning("No XML files found to parse")
        return 0

    # Initialize parser and client
    parser = LegalDocumentParser()
    eurlex_client = EURLexClient()

    parsed_documents = []

    for xml_file in xml_files[:10]:  # Process up to 10 documents
        try:
            logger.info(f"Parsing {xml_file.name}...")

            # Extract CELEX from filename
            celex = xml_file.stem  # Remove .xml extension

            # Get regulation metadata
            try:
                regulation = eurlex_client.extract_celex_metadata(celex)
            except Exception as e:
                logger.warning(f"Could not fetch metadata for {celex}: {e}")
                regulation = None

            # Parse document
            document = parser.parse(xml_file, regulation=regulation)

            # Save parsed document
            import pickle

            output_path = processed_dir / f"{celex}.pkl"
            with output_path.open("wb") as f:
                pickle.dump(document, f)

            parsed_documents.append(
                {
                    "celex": celex,
                    "title": document.regulation.full_title,
                    "chapters": len(document.chapters),
                    "path": str(output_path),
                }
            )

            logger.info(
                f"Parsed {celex}: {len(document.chapters)} chapters, "
                f"{sum(len(ch.articles) for ch in document.chapters)} articles"
            )

        except Exception as e:
            logger.error(f"Failed to parse {xml_file.name}: {str(e)}")
            continue

    eurlex_client.close()

    # Push to XCom
    context["ti"].xcom_push(key="parsed_documents", value=parsed_documents)

    logger.info(f"Successfully parsed {len(parsed_documents)} documents")
    return len(parsed_documents)


def chunk_documents(**context):
    """
    Chunk parsed documents using legal-aware strategy.

    Preserves article boundaries and adds metadata.
    """
    import pickle

    from services.data_pipeline.src.chunking import LegalChunker
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Starting document chunking...")

    # Get parsed documents from previous task
    ti = context["ti"]
    parsed_docs = ti.xcom_pull(key="parsed_documents", task_ids="parse_documents")

    if not parsed_docs:
        logger.warning("No parsed documents to chunk")
        return 0

    # Initialize chunker with tiktoken for OpenAI compatibility
    chunker = LegalChunker(
        max_chunk_tokens=512,
        overlap_sentences=2,
        tokenizer_type="tiktoken",
        encoding_name="cl100k_base",
    )

    all_chunks = []
    chunks_dir = Path("/opt/airflow/data/processed/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for doc_info in parsed_docs:
        try:
            celex = doc_info["celex"]
            doc_path = Path(doc_info["path"])

            logger.info(f"Chunking {celex}...")

            # Load parsed document
            with doc_path.open("rb") as f:
                document = pickle.load(f)

            # Chunk document
            chunks = chunker.chunk_document(document)

            # Save chunks
            chunks_path = chunks_dir / f"{celex}_chunks.pkl"
            with chunks_path.open("wb") as f:
                pickle.dump(chunks, f)

            all_chunks.append(
                {
                    "celex": celex,
                    "num_chunks": len(chunks),
                    "path": str(chunks_path),
                }
            )

            logger.info(f"Created {len(chunks)} chunks from {celex}")

        except Exception as e:
            logger.error(f"Failed to chunk {doc_info['celex']}: {str(e)}")
            continue

    # Push to XCom
    ti.xcom_push(key="chunked_documents", value=all_chunks)

    total_chunks = sum(doc["num_chunks"] for doc in all_chunks)
    logger.info(f"Created total of {total_chunks} chunks from {len(all_chunks)} documents")

    return total_chunks


def generate_embeddings(**context):
    """
    Generate embeddings for all chunks.

    Uses OpenAI text-embedding-3-large model via API.
    """
    import pickle

    from services.data_pipeline.src.embeddings import EmbeddingGenerator
    from shared.config import get_settings
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Starting embedding generation...")

    settings = get_settings()

    # Get chunked documents from previous task
    ti = context["ti"]
    chunked_docs = ti.xcom_pull(key="chunked_documents", task_ids="chunk_documents")

    if not chunked_docs:
        logger.warning("No chunked documents to embed")
        return 0

    # Initialize embedding generator with OpenAI
    generator = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=100,  # OpenAI supports larger batches
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    logger.info(f"Using embedding model: {generator.model_name}")
    logger.info(f"Embedding dimension: {generator.embedding_dim}")

    embedded_docs = []
    embeddings_dir = Path("/opt/airflow/data/processed/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    for doc_info in chunked_docs:
        try:
            celex = doc_info["celex"]
            chunks_path = Path(doc_info["path"])

            logger.info(f"Generating embeddings for {celex} ({doc_info['num_chunks']} chunks)...")

            # Load chunks
            with chunks_path.open("rb") as f:
                chunks = pickle.load(f)

            # Generate embeddings
            chunks_with_embeddings = generator.generate_embeddings(chunks, normalize=True)

            # Save embedded chunks
            embedded_path = embeddings_dir / f"{celex}_embedded.pkl"
            with embedded_path.open("wb") as f:
                pickle.dump(chunks_with_embeddings, f)

            embedded_docs.append(
                {
                    "celex": celex,
                    "num_chunks": len(chunks_with_embeddings),
                    "path": str(embedded_path),
                }
            )

            logger.info(f"Generated {len(chunks_with_embeddings)} embeddings for {celex}")

        except Exception as e:
            logger.error(f"Failed to generate embeddings for {doc_info['celex']}: {str(e)}")
            continue

    # Push to XCom
    ti.xcom_push(key="embedded_documents", value=embedded_docs)

    total_embeddings = sum(doc["num_chunks"] for doc in embedded_docs)
    logger.info(f"Generated {total_embeddings} embeddings for {len(embedded_docs)} documents")

    return total_embeddings


def index_to_qdrant(**context):
    """
    Index embedded chunks to Qdrant vector database.

    Creates collection if needed and batch indexes all chunks.
    """
    import pickle

    from services.data_pipeline.src.indexing import QdrantIndexer
    from shared.config import get_settings
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Starting Qdrant indexing...")

    settings = get_settings()

    # Get embedded documents from previous task
    ti = context["ti"]
    embedded_docs = ti.xcom_pull(key="embedded_documents", task_ids="generate_embeddings")

    if not embedded_docs:
        logger.warning("No embedded documents to index")
        return 0

    # Initialize indexer
    indexer = QdrantIndexer(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        embedding_dim=settings.embedding_dimension,
    )

    # Create collection if it doesn't exist
    try:
        indexer.create_collection(recreate=False)
    except Exception as e:
        logger.warning(f"Collection already exists or failed to create: {e}")

    # Index all chunks
    total_indexed = 0

    for doc_info in embedded_docs:
        try:
            celex = doc_info["celex"]
            embedded_path = Path(doc_info["path"])

            logger.info(f"Indexing {celex} to Qdrant...")

            # Load embedded chunks
            with embedded_path.open("rb") as f:
                chunks = pickle.load(f)

            # Delete existing chunks for this document (for re-indexing)
            try:
                indexer.delete_by_celex(celex)
            except Exception as e:
                logger.warning(f"Could not delete existing chunks for {celex}: {e}")

            # Index chunks
            indexed = indexer.index_chunks(chunks, batch_size=100, show_progress=True)
            total_indexed += indexed

            logger.info(f"Indexed {indexed} chunks for {celex}")

        except Exception as e:
            logger.error(f"Failed to index {doc_info['celex']}: {str(e)}")
            continue

    # Get collection info
    try:
        collection_info = indexer.get_collection_info()
        logger.info(f"Qdrant collection info:")
        for key, value in collection_info.items():
            logger.info(f"  {key}: {value}")
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")

    indexer.close()

    logger.info(f"Successfully indexed {total_indexed} chunks to Qdrant")
    return total_indexed


def validate_indexing(**context):
    """
    Validate that indexing was successful.

    Performs test queries to verify retrieval works.
    """
    from services.data_pipeline.src.indexing import QdrantIndexer
    from shared.config import get_settings
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Validating Qdrant indexing...")

    settings = get_settings()

    # Initialize indexer
    indexer = QdrantIndexer(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        embedding_dim=settings.embedding_dimension,
    )

    # Get collection stats
    try:
        info = indexer.get_collection_info()
        logger.info(f"Collection has {info['vectors_count']} vectors")

        if info["vectors_count"] == 0:
            logger.error("No vectors in collection!")
            return False

        # Count by regulation
        filters = {"regulation_name": "GDPR"}
        gdpr_count = indexer.count_documents(filters)
        logger.info(f"GDPR chunks: {gdpr_count}")

        logger.info("Indexing validation successful")
        return True

    except Exception as e:
        logger.error(f"Indexing validation failed: {str(e)}")
        return False
    finally:
        indexer.close()


# Define DAG
with DAG(
    "document_processing",
    default_args=default_args,
    description="Process legal documents: parse, chunk, embed, and index",
    schedule_interval=None,  # Triggered by other DAGs
    start_date=days_ago(1),
    catchup=False,
    tags=["data-processing", "legal-documents", "qdrant"],
) as dag:

    # Task 1: Parse documents
    task_parse = PythonOperator(
        task_id="parse_documents",
        python_callable=parse_documents,
        provide_context=True,
    )

    # Task 2: Chunk documents
    task_chunk = PythonOperator(
        task_id="chunk_documents",
        python_callable=chunk_documents,
        provide_context=True,
    )

    # Task 3: Generate embeddings
    task_embed = PythonOperator(
        task_id="generate_embeddings",
        python_callable=generate_embeddings,
        provide_context=True,
    )

    # Task 4: Index to Qdrant
    task_index = PythonOperator(
        task_id="index_to_qdrant",
        python_callable=index_to_qdrant,
        provide_context=True,
    )

    # Task 5: Validate indexing
    task_validate = PythonOperator(
        task_id="validate_indexing",
        python_callable=validate_indexing,
        provide_context=True,
    )

    # Define task dependencies
    task_parse >> task_chunk >> task_embed >> task_index >> task_validate
