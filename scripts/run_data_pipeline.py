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
from data_pipeline_logging_utils import PipelineStageLogger
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
    import time

    pipeline_start_time = time.time()

    args = _parse_args()
    settings = get_settings()
    logger = get_logger(__name__)

    # Visual pipeline start
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║              STARTING DATA PIPELINE EXECUTION                     ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")

    logger.info(
        "Pipeline configuration",
        extra={
            "limit": args.limit,
            "start_date": args.start_date,
            "recreate_collection": args.recreate_collection,
            "download_format": args.download_format,
            "eurlex_timeout": args.eurlex_timeout,
            "eurlex_retries": args.eurlex_retries,
        },
    )

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

    logger.debug(
        "Data directories initialized",
        extra={
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
            "chunks_dir": str(chunks_dir),
            "embeddings_dir": str(embeddings_dir),
        },
    )

    # Stage 1: Document Discovery and Download
    download_stage = PipelineStageLogger("document_download", "download")
    download_stage.log_start(limit=args.limit, start_date=start_date.isoformat())
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
        download_stage.log_warning("No CELEX IDs found")
        download_stage.log_complete(documents_found=0)
        eurlex_client.close()
        return 0

    logger.info(f"Will process {len(celex_ids)} documents: {', '.join(celex_ids)}")

    downloaded_paths: list[Path] = []
    download_errors = 0

    for idx, celex in enumerate(celex_ids, 1):
        output_path = raw_dir / f"{celex}.{args.download_format}"
        download_stage.log_info(f"Downloading {idx}/{len(celex_ids)}: {celex}")

        try:
            doc_start = time.time()
            eurlex_client.download_document_to_file(
                celex=celex,
                output_path=output_path,
                format=args.download_format,
                language="EN",
            )
            doc_duration = (time.time() - doc_start) * 1000
            file_size = output_path.stat().st_size

            download_stage.log_info(
                f"✓ Downloaded {celex} ({file_size:,} bytes, {doc_duration:.0f}ms)"
            )
            downloaded_paths.append(output_path)
        except Exception as exc:
            logger.error(f"Failed to download {celex}: {exc}")
            download_errors += 1

    download_stage.log_complete(
        documents_downloaded=len(downloaded_paths),
        download_errors=download_errors,
    )

    if not downloaded_paths:
        existing_raw = list(raw_dir.glob("*.xml")) + list(raw_dir.glob("*.html"))
        if existing_raw:
            logger.warning("No documents downloaded; using existing raw XML files.")
            logger.info(f"Found {len(existing_raw)} existing raw files")
            downloaded_paths = existing_raw
        else:
            logger.warning("No documents downloaded and no existing files found.")
            eurlex_client.close()
            return 0

    # Stage 2: Document Parsing
    parse_stage = PipelineStageLogger("document_parsing", "parse")
    parse_stage.log_start(documents_count=len(downloaded_paths))
    logger.info("Parsing downloaded documents...")
    parser = LegalDocumentParser()
    parsed_docs: list[dict[str, str]] = []
    parse_errors = 0
    total_chapters = 0
    total_articles = 0

    for idx, doc_path in enumerate(downloaded_paths, 1):
        celex = doc_path.stem
        parse_stage.log_info(f"Parsing {idx}/{len(downloaded_paths)}: {celex}")

        try:
            parse_start = time.time()

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

            parse_duration = (time.time() - parse_start) * 1000

            chapters_count = len(document.chapters)
            articles_count = sum(len(ch.articles) for ch in document.chapters)
            total_chapters += chapters_count
            total_articles += articles_count

            parse_stage.log_info(
                f"✓ Parsed {celex}: {chapters_count} chapters, {articles_count} articles ({parse_duration:.0f}ms)"
            )

            parsed_path = processed_dir / f"{celex}.pkl"
            with parsed_path.open("wb") as handle:
                pickle.dump(document, handle)

            parsed_docs.append({"celex": celex, "path": str(parsed_path)})
        except Exception as exc:
            logger.error(f"Failed to parse {celex}: {exc}")
            parse_errors += 1

    eurlex_client.close()

    parse_stage.log_complete(
        documents_parsed=len(parsed_docs),
        parse_errors=parse_errors,
        total_chapters=total_chapters,
        total_articles=total_articles,
    )

    if not parsed_docs:
        logger.warning("No documents parsed.")
        return 0

    # Stage 3: Document Chunking
    chunk_stage = PipelineStageLogger("document_chunking", "chunk")
    chunk_stage.log_start(documents_count=len(parsed_docs))
    logger.info("Chunking parsed documents...")
    chunker = LegalChunker(
        max_chunk_tokens=settings.chunk_size,
        overlap_sentences=2,
        tokenizer_type="tiktoken",
        encoding_name="cl100k_base",
    )

    chunked_docs: list[dict[str, str]] = []
    chunk_errors = 0
    total_chunks_created = 0
    total_chunks_filtered = 0

    for idx, doc_info in enumerate(parsed_docs, 1):
        celex = doc_info["celex"]
        chunk_stage.log_info(f"Chunking {idx}/{len(parsed_docs)}: {celex}")

        try:
            chunk_start = time.time()

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
            total_chunks_filtered += filtered_count
            total_chunks_created += len(chunks)

            chunk_duration = (time.time() - chunk_start) * 1000
            avg_chunk_length = sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0

            chunk_stage.log_info(
                f"✓ Chunked {celex}: {len(chunks)} chunks (filtered: {filtered_count}, avg length: {avg_chunk_length:.0f} chars, {chunk_duration:.0f}ms)"
            )

            if filtered_count > 0:
                chunk_stage.log_debug(f"Filtered out {filtered_count} problematic chunks from {celex}")

            if not chunks:
                logger.warning(f"No valid chunks for {celex} after filtering")
                continue

            chunks_path = chunks_dir / f"{celex}_chunks.pkl"
            with chunks_path.open("wb") as handle:
                pickle.dump(chunks, handle)

            chunked_docs.append({"celex": celex, "path": str(chunks_path)})
        except Exception as exc:
            logger.error(f"Failed to chunk {celex}: {exc}")
            chunk_errors += 1

    chunk_stage.log_complete(
        documents_chunked=len(chunked_docs),
        chunk_errors=chunk_errors,
        total_chunks_created=total_chunks_created,
        total_chunks_filtered=total_chunks_filtered,
        avg_chunks_per_doc=total_chunks_created / len(chunked_docs) if chunked_docs else 0,
    )

    if not chunked_docs:
        logger.warning("No documents chunked.")
        return 0

    # Stage 4: Embedding Generation
    embed_stage = PipelineStageLogger("embedding_generation", "embed")
    embed_stage.log_start(documents_count=len(chunked_docs))
    logger.info("Generating embeddings...")
    generator = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=50,  # Reduced from 100 to prevent API errors
        show_progress=True,
        dimensions=settings.embedding_dimension,
    )

    embedded_docs: list[Path] = []
    embed_errors = 0
    total_embeddings_generated = 0
    total_embeddings_skipped = 0

    for idx, doc_info in enumerate(chunked_docs, 1):
        celex = doc_info["celex"]
        embedded_path = embeddings_dir / f"{celex}_embedded.pkl"

        # Skip if embeddings already exist and are valid
        if embedded_path.exists():
            try:
                with embedded_path.open("rb") as handle:
                    existing_chunks = pickle.load(handle)
                if existing_chunks and len(existing_chunks) > 0:
                    embed_stage.log_info(f"Using existing embeddings for {celex} ({len(existing_chunks)} chunks)")
                    total_embeddings_skipped += len(existing_chunks)
                    embedded_docs.append(embedded_path)
                    continue
            except Exception:
                logger.warning(f"Existing embeddings for {celex} are invalid, regenerating...")

        try:
            embed_start = time.time()

            with Path(doc_info["path"]).open("rb") as handle:
                chunks = pickle.load(handle)

            if not chunks:
                logger.warning(f"No chunks to embed for {celex}")
                continue

            embed_stage.log_info(f"Embedding {idx}/{len(chunked_docs)}: {celex} ({len(chunks)} chunks)")
            embedded_chunks = generator.generate_embeddings(chunks, normalize=True)

            embed_duration = (time.time() - embed_start) * 1000

            if not embedded_chunks or len(embedded_chunks) == 0:
                logger.error(f"Embedding generation returned empty list for {celex}")
                embed_errors += 1
                continue

            total_embeddings_generated += len(embedded_chunks)
            throughput = len(embedded_chunks) / (embed_duration / 1000) if embed_duration > 0 else 0

            embed_stage.log_info(
                f"✓ Embedded {celex}: {len(embedded_chunks)} embeddings ({embed_duration:.0f}ms, {throughput:.1f} emb/sec)"
            )

            with embedded_path.open("wb") as handle:
                pickle.dump(embedded_chunks, handle)

            embedded_docs.append(embedded_path)
        except Exception as exc:
            logger.error(f"Failed to embed {celex}: {exc}")
            logger.error("You can re-run the pipeline to retry this document")
            embed_errors += 1

    embed_stage.log_complete(
        documents_embedded=len(embedded_docs),
        embed_errors=embed_errors,
        total_embeddings_generated=total_embeddings_generated,
        total_embeddings_skipped=total_embeddings_skipped,
    )

    if not embedded_docs:
        logger.warning("No embeddings generated.")
        return 0

    # Stage 5: Indexing to Qdrant
    index_stage = PipelineStageLogger("qdrant_indexing", "index")
    index_stage.log_start(documents_count=len(embedded_docs))
    logger.info("Indexing embeddings into Qdrant...")
    indexer = QdrantIndexer()
    indexer.create_collection(recreate=args.recreate_collection)

    total_indexed = 0
    index_errors = 0

    for idx, embedded_path in enumerate(embedded_docs, 1):
        celex = embedded_path.stem.replace("_embedded", "")
        index_stage.log_info(f"Indexing {idx}/{len(embedded_docs)}: {celex}")

        try:
            index_start = time.time()

            with embedded_path.open("rb") as handle:
                embedded_chunks = pickle.load(handle)

            indexed = indexer.index_chunks(embedded_chunks, batch_size=100, show_progress=True)
            index_duration = (time.time() - index_start) * 1000
            throughput = indexed / (index_duration / 1000) if index_duration > 0 else 0

            index_stage.log_info(
                f"✓ Indexed {celex}: {indexed} points ({index_duration:.0f}ms, {throughput:.1f} pts/sec)"
            )

            total_indexed += indexed
        except Exception as exc:
            logger.error(f"Failed to index {embedded_path.name}: {exc}")
            index_errors += 1

    index_stage.log_complete(
        documents_indexed=len(embedded_docs) - index_errors,
        index_errors=index_errors,
        total_points_indexed=total_indexed,
    )

    indexer.close()

    # Pipeline completion
    pipeline_duration = (time.time() - pipeline_start_time) * 1000

    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║            DATA PIPELINE COMPLETED SUCCESSFULLY                   ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")

    logger.log_performance(
        operation="data_pipeline_execution",
        duration_ms=pipeline_duration,
        documents_processed=len(celex_ids),
        documents_downloaded=len(downloaded_paths),
        documents_parsed=len(parsed_docs),
        documents_chunked=len(chunked_docs),
        documents_embedded=len(embedded_docs),
        total_chunks_created=total_chunks_created,
        total_embeddings_generated=total_embeddings_generated,
        total_points_indexed=total_indexed,
    )

    logger.info(
        "Pipeline summary",
        extra={
            "total_duration_ms": pipeline_duration,
            "total_duration_sec": pipeline_duration / 1000,
            "documents_processed": len(celex_ids),
            "download_errors": download_errors,
            "parse_errors": parse_errors,
            "chunk_errors": chunk_errors,
            "embed_errors": embed_errors,
            "index_errors": index_errors,
            "success_rate": (len(embedded_docs) - index_errors) / len(celex_ids) if celex_ids else 0,
            "total_chapters": total_chapters,
            "total_articles": total_articles,
            "total_chunks_created": total_chunks_created,
            "total_chunks_filtered": total_chunks_filtered,
            "total_embeddings_generated": total_embeddings_generated,
            "total_embeddings_skipped": total_embeddings_skipped,
            "total_points_indexed": total_indexed,
        },
    )

    logger.log_audit(
        action="data_pipeline_completed",
        resource="eu_legal_documents",
        result="success",
        processing_time_ms=pipeline_duration,
        documents_indexed=len(embedded_docs) - index_errors,
        total_points=total_indexed,
    )

    logger.info(f"✓ Pipeline complete. Indexed {total_indexed} chunks in {pipeline_duration/1000:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
