"""
EUR-Lex Daily Ingestion DAG

Fetches latest EU regulations from EUR-Lex and triggers document processing.
"""

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from airflow import DAG

# Default arguments
default_args = {
    "owner": "conformai",
    "depends_on_past": False,
    "email": ["admin@conformai.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def fetch_recent_documents(**context):
    """
    Fetch recent documents from EUR-Lex API.

    Query parameters:
    - Document types: Regulation, Directive
    - Date range: Last 7 days
    - Subjects: AI, data protection
    """
    import httpx

    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Fetching recent documents from EUR-Lex...")

    # EUR-Lex SPARQL endpoint
    sparql_endpoint = "https://publications.europa.eu/webapi/rdf/sparql"

    # SPARQL query for recent AI and data protection regulations
    query = """
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?work ?celex ?title ?date
    WHERE {
        ?work cdm:work_has_resource-type ?type .
        ?work cdm:resource_legal_id_celex ?celex .
        ?work cdm:work_title_label ?title .
        ?work cdm:work_date_document ?date .

        FILTER (
            CONTAINS(LCASE(str(?title)), "artificial intelligence") ||
            CONTAINS(LCASE(str(?title)), "data protection") ||
            CONTAINS(LCASE(str(?title)), "gdpr")
        )

        FILTER (?date >= "2024-01-01"^^xsd:date)
    }
    ORDER BY DESC(?date)
    LIMIT 50
    """

    try:
        response = httpx.post(
            sparql_endpoint,
            headers={"Accept": "application/sparql-results+json"},
            data={"query": query},
            timeout=60.0,
        )
        response.raise_for_status()

        results = response.json()
        bindings = results.get("results", {}).get("bindings", [])

        logger.info(f"Found {len(bindings)} documents")

        # Push document list to XCom for next task
        document_list = [
            {
                "celex": item["celex"]["value"],
                "title": item["title"]["value"],
                "date": item["date"]["value"],
                "url": item["work"]["value"],
            }
            for item in bindings
        ]

        context["ti"].xcom_push(key="document_list", value=document_list)
        return len(document_list)

    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        raise


def detect_changes(**context):
    """
    Detect which documents are new or updated.

    Compares fetched documents against database to identify changes.
    """
    from datetime import date

    from sqlalchemy import create_engine, text

    from shared.config.settings import get_settings
    from shared.utils import get_logger

    logger = get_logger(__name__)

    # Pull document list from previous task
    ti = context["ti"]
    document_list = ti.xcom_pull(key="document_list", task_ids="fetch_recent_documents")

    logger.info(f"Checking {len(document_list)} documents for changes...")

    if not document_list:
        logger.info("No documents to check")
        ti.xcom_push(key="new_documents", value=[])
        return 0

    def _normalize_date(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            return datetime.fromisoformat(str(value)).date()
        except ValueError:
            return None

    celex_ids = [doc.get("celex") for doc in document_list if doc.get("celex")]
    if not celex_ids:
        logger.info("No CELEX identifiers found in document list")
        ti.xcom_push(key="new_documents", value=[])
        return 0

    settings = get_settings()
    engine = create_engine(settings.database_url)

    query = text(
        """
        SELECT celex, document_date
        FROM eurlex_documents
        WHERE celex = ANY(:celex_ids)
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query, {"celex_ids": celex_ids}).fetchall()

    existing_by_celex = {row[0]: row[1] for row in rows}

    new_documents = []
    for doc in document_list:
        celex = doc.get("celex")
        if not celex:
            continue
        existing_date = _normalize_date(existing_by_celex.get(celex))
        fetched_date = _normalize_date(doc.get("date"))

        if existing_date is None:
            new_documents.append(doc)
            continue
        if fetched_date is None:
            continue
        if fetched_date > existing_date:
            new_documents.append(doc)

    logger.info(f"Found {len(new_documents)} new/updated documents")

    ti.xcom_push(key="new_documents", value=new_documents)
    return len(new_documents)


def download_documents(**context):
    """
    Download full document content (XML/PDF).

    Downloads from EUR-Lex formex endpoints.
    """
    from pathlib import Path

    import httpx

    from shared.utils import get_logger

    logger = get_logger(__name__)

    ti = context["ti"]
    new_documents = ti.xcom_pull(key="new_documents", task_ids="detect_changes")

    if not new_documents:
        logger.info("No new documents to download")
        return 0

    # Create download directory
    download_dir = Path("/opt/airflow/data/raw")
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0

    for doc in new_documents[:5]:  # Limit to 5 for testing
        celex = doc["celex"]
        logger.info(f"Downloading {celex}...")

        # Construct EUR-Lex download URL (Formex XML)
        # Format: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}&format=XML
        url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}&format=XML"

        try:
            response = httpx.get(url, timeout=60.0, follow_redirects=True)
            response.raise_for_status()

            # Save to file
            file_path = download_dir / f"{celex}.xml"
            file_path.write_bytes(response.content)

            logger.info(f"Saved {celex} to {file_path}")
            downloaded_count += 1

        except Exception as e:
            logger.error(f"Failed to download {celex}: {str(e)}")
            continue

    logger.info(f"Downloaded {downloaded_count} documents")
    return downloaded_count


def store_raw_documents(**context):
    """
    Store raw documents in object storage (S3/MinIO).

    Optional: Upload to S3-compatible storage for backup.
    """
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Storing raw documents in object storage...")

    # TODO: Implement S3/MinIO upload
    # For now, files are stored locally in /opt/airflow/data/raw

    return True


def trigger_processing_dag(**context):
    """
    Trigger document processing DAG.

    Initiates parsing, chunking, and indexing pipeline.
    """
    from shared.utils import get_logger

    logger = get_logger(__name__)
    logger.info("Triggering document processing DAG...")

    # TODO: Trigger processing DAG via Airflow API or TriggerDagRunOperator

    return True


# Define DAG
with DAG(
    "eurlex_daily_ingestion",
    default_args=default_args,
    description="Fetch latest EU regulations from EUR-Lex",
    schedule_interval="0 2 * * *",  # 2 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=["data-ingestion", "eurlex", "regulations"],
) as dag:

    # Task 1: Fetch recent documents
    task_fetch = PythonOperator(
        task_id="fetch_recent_documents",
        python_callable=fetch_recent_documents,
        provide_context=True,
    )

    # Task 2: Detect changes
    task_detect = PythonOperator(
        task_id="detect_changes",
        python_callable=detect_changes,
        provide_context=True,
    )

    # Task 3: Download documents
    task_download = PythonOperator(
        task_id="download_documents",
        python_callable=download_documents,
        provide_context=True,
    )

    # Task 4: Store raw documents
    task_store = PythonOperator(
        task_id="store_raw_documents",
        python_callable=store_raw_documents,
        provide_context=True,
    )

    # Task 5: Trigger processing
    task_trigger = PythonOperator(
        task_id="trigger_processing_dag",
        python_callable=trigger_processing_dag,
        provide_context=True,
    )

    # Define task dependencies
    task_fetch >> task_detect >> task_download >> task_store >> task_trigger
