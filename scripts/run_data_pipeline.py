#!/usr/bin/env python3
"""
Run the ConformAI data pipeline end-to-end.

Steps:
1. Fetch sample EU AI Act-related documents from EUR-Lex
2. Parse and chunk them
3. Generate embeddings
4. Index into Qdrant
"""

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

# Add project root and data-pipeline src to path
project_root = Path(__file__).parent.parent
data_pipeline_root = project_root / "services" / "data-pipeline" / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(data_pipeline_root))

from chunking import LegalChunker
from clients import EURLexClient
from embeddings import EmbeddingGenerator
from indexing import QdrantIndexer
from parsers import LegalDocumentParser
from shared.config import get_settings
from shared.utils import get_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ConformAI data pipeline.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of EUR-Lex documents to fetch.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Filter documents from this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Recreate Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--celex",
        action="append",
        default=[],
        help="CELEX ID to process (repeatable). Skips EUR-Lex search.",
    )
    parser.add_argument(
        "--download-format",
        choices=["xml", "html"],
        default="xml",
        help="Preferred EUR-Lex download format.",
    )
    parser.add_argument(
        "--eurlex-timeout",
        type=int,
        default=120,
        help="EUR-Lex request timeout in seconds.",
    )
    parser.add_argument(
        "--eurlex-retries",
        type=int,
        default=5,
        help="EUR-Lex max retries for transient failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    logger = get_logger(__name__)

    try:
        start_date = date.fromisoformat(args.start_date)
    except ValueError:
        logger.error("Invalid --start-date format. Use YYYY-MM-DD.")
        return 1

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    chunks_dir = processed_dir / "chunks"
    embeddings_dir = data_dir / "embeddings"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching AI-related EUR-Lex documents...")
    eurlex_client = EURLexClient(
        sparql_endpoint=settings.eurlex_api_base_url,
        timeout=args.eurlex_timeout,
        max_retries=args.eurlex_retries,
    )

    fallback_celexes = [
        "32016R0679",  # GDPR
        "52021PC0206",  # AI Act proposal
        "52020PC0767",  # AI Act annexes
    ]

    if args.celex:
        celex_ids = args.celex
        logger.info(f"Using provided CELEX IDs: {', '.join(celex_ids)}")
    else:
        try:
            docs = eurlex_client.search_ai_related_documents(
                start_date=start_date, limit=args.limit
            )
            docs = [doc for doc in docs if doc.get("celex")]
            celex_ids = list({doc["celex"] for doc in docs})
        except Exception as exc:
            logger.warning(f"EUR-Lex search failed, using fallback CELEX list: {exc}")
            celex_ids = fallback_celexes

    if not celex_ids:
        logger.warning("No CELEX IDs available to process.")
        eurlex_client.close()
        return 0

    downloaded_paths: list[Path] = []
    for celex in celex_ids:
        output_path = raw_dir / f"{celex}.{args.download_format}"
        try:
            eurlex_client.download_document_to_file(
                celex=celex,
                output_path=output_path,
                format=args.download_format,
                language="EN",
            )
            downloaded_paths.append(output_path)
        except Exception as exc:
            logger.error(f"Failed to download {celex}: {exc}")

    if not downloaded_paths:
        existing_raw = list(raw_dir.glob("*.xml")) + list(raw_dir.glob("*.html"))
        if existing_raw:
            logger.warning("No documents downloaded; using existing raw XML files.")
            downloaded_paths = existing_raw
        else:
            logger.warning("No documents downloaded.")
            eurlex_client.close()
            return 0

    logger.info("Parsing downloaded documents...")
    parser = LegalDocumentParser()
    parsed_docs: list[dict[str, str]] = []

    for doc_path in downloaded_paths:
        celex = doc_path.stem
        try:
            try:
                regulation = eurlex_client.extract_celex_metadata(celex)
            except Exception as exc:
                logger.warning(f"Metadata lookup failed for {celex}: {exc}")
                regulation = None

            try:
                document = parser.parse(doc_path, regulation=regulation)
            except Exception as exc:
                logger.warning(f"Primary parse failed for {celex}: {exc}")
                document = parser.parse_html(doc_path, regulation=regulation)
            parsed_path = processed_dir / f"{celex}.pkl"
            with parsed_path.open("wb") as handle:
                pickle.dump(document, handle)
            parsed_docs.append({"celex": celex, "path": str(parsed_path)})
        except Exception as exc:
            logger.error(f"Failed to parse {celex}: {exc}")

    eurlex_client.close()

    if not parsed_docs:
        logger.warning("No documents parsed.")
        return 0

    logger.info("Chunking parsed documents...")
    chunker = LegalChunker(
        max_chunk_tokens=settings.chunk_size,
        overlap_sentences=2,
        tokenizer_type="tiktoken",
        encoding_name="cl100k_base",
    )

    chunked_docs: list[dict[str, str]] = []
    for doc_info in parsed_docs:
        celex = doc_info["celex"]
        try:
            with Path(doc_info["path"]).open("rb") as handle:
                document = pickle.load(handle)

            chunks = chunker.chunk_document(document)

            # Filter out problematic chunks (empty or too short)
            original_count = len(chunks)
            chunks = [
                chunk for chunk in chunks
                if hasattr(chunk, 'text') and chunk.text and len(chunk.text.strip()) >= 10
            ]

            filtered_count = original_count - len(chunks)
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} problematic chunks from {celex}")

            if not chunks:
                logger.warning(f"No valid chunks for {celex} after filtering")
                continue

            chunks_path = chunks_dir / f"{celex}_chunks.pkl"
            with chunks_path.open("wb") as handle:
                pickle.dump(chunks, handle)

            chunked_docs.append({"celex": celex, "path": str(chunks_path)})
        except Exception as exc:
            logger.error(f"Failed to chunk {celex}: {exc}")

    if not chunked_docs:
        logger.warning("No documents chunked.")
        return 0

    logger.info("Generating embeddings...")
    generator = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=50,  # Reduced from 100 to prevent API errors
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    embedded_docs: list[Path] = []
    for doc_info in chunked_docs:
        celex = doc_info["celex"]
        embedded_path = embeddings_dir / f"{celex}_embedded.pkl"

        # Skip if embeddings already exist and are valid
        if embedded_path.exists():
            try:
                with embedded_path.open("rb") as handle:
                    existing_chunks = pickle.load(handle)
                if existing_chunks and len(existing_chunks) > 0:
                    logger.info(f"Using existing embeddings for {celex} ({len(existing_chunks)} chunks)")
                    embedded_docs.append(embedded_path)
                    continue
            except Exception:
                logger.warning(f"Existing embeddings for {celex} are invalid, regenerating...")

        try:
            with Path(doc_info["path"]).open("rb") as handle:
                chunks = pickle.load(handle)

            if not chunks:
                logger.warning(f"No chunks to embed for {celex}")
                continue

            logger.info(f"Embedding {len(chunks)} chunks for {celex}...")
            embedded_chunks = generator.generate_embeddings(chunks, normalize=True)

            if not embedded_chunks or len(embedded_chunks) == 0:
                logger.error(f"Embedding generation returned empty list for {celex}")
                continue

            with embedded_path.open("wb") as handle:
                pickle.dump(embedded_chunks, handle)

            logger.info(f"✓ Saved {len(embedded_chunks)} embeddings for {celex}")
            embedded_docs.append(embedded_path)
        except Exception as exc:
            logger.error(f"Failed to embed {celex}: {exc}")
            logger.error("You can re-run the pipeline to retry this document")

    if not embedded_docs:
        logger.warning("No embeddings generated.")
        return 0

    logger.info("Indexing embeddings into Qdrant...")
    indexer = QdrantIndexer()
    indexer.create_collection(recreate=args.recreate_collection)

    total_indexed = 0
    for embedded_path in embedded_docs:
        try:
            with embedded_path.open("rb") as handle:
                embedded_chunks = pickle.load(handle)
            indexed = indexer.index_chunks(embedded_chunks, batch_size=100, show_progress=True)
            total_indexed += indexed
        except Exception as exc:
            logger.error(f"Failed to index {embedded_path.name}: {exc}")

    indexer.close()
    logger.info(f"Pipeline complete. Indexed {total_indexed} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
