#!/usr/bin/env python3
"""
Test Data Pipeline Components

Tests each component of the data pipeline:
1. EUR-Lex API client
2. Legal document parser
3. Legal-aware chunker
4. Embedding generator
5. Qdrant indexer

Run this script to validate the pipeline before running Airflow DAGs.
"""

import sys
from datetime import date
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_eurlex_client():
    """Test EUR-Lex API client."""
    print("=" * 80)
    print("TEST 1: EUR-Lex Client")
    print("=" * 80)

    from services.data_pipeline.src.clients import EURLexClient

    client = EURLexClient()

    try:
        # Test 1: Search for AI-related documents
        print("\n1. Searching for AI-related documents...")
        docs = client.search_ai_related_documents(start_date=date(2023, 1, 1), limit=5)
        print(f"   Found {len(docs)} documents")

        if docs:
            print(f"   Sample: {docs[0]['celex']} - {docs[0]['title'][:60]}...")

        # Test 2: Get GDPR metadata
        print("\n2. Fetching GDPR metadata...")
        gdpr = client.get_gdpr_document()
        print(f"   CELEX: {gdpr['celex']}")
        print(f"   Title: {gdpr['title'][:60]}...")

        # Test 3: Download sample document (commented out to avoid large download)
        # print("\n3. Downloading GDPR (XML)...")
        # content = client.download_document("32016R0679", format="xml")
        # print(f"   Downloaded {len(content)} bytes")

        print("\n✅ EUR-Lex client tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ EUR-Lex client tests failed: {str(e)}")
        return False
    finally:
        client.close()


def test_legal_parser():
    """Test legal document parser."""
    print("\n" + "=" * 80)
    print("TEST 2: Legal Document Parser")
    print("=" * 80)

    from services.data_pipeline.src.parsers import LegalDocumentParser


    parser = LegalDocumentParser()

    try:
        # Create a sample HTML document for testing
        sample_html = """
        <html>
        <head><title>Sample Regulation</title></head>
        <body>
            <h2 class="ti-chapter">CHAPTER I - General Provisions</h2>
            <h3 class="ti-art">Article 1 - Subject matter</h3>
            <p>This regulation establishes rules for AI systems.</p>
            <h3 class="ti-art">Article 2 - Scope</h3>
            <p>This regulation applies to all AI systems placed on the market.</p>
        </body>
        </html>
        """

        # Save to temp file
        temp_dir = Path("/tmp/conformai_test")
        temp_dir.mkdir(exist_ok=True)
        html_path = temp_dir / "sample.html"
        html_path.write_text(sample_html)

        print("\n1. Parsing HTML document...")
        document = parser.parse(html_path)
        print(f"   Chapters: {len(document.chapters)}")
        print(f"   Total articles: {sum(len(ch.articles) for ch in document.chapters)}")

        if document.chapters:
            chapter = document.chapters[0]
            print(f"   First chapter: {chapter.number} - {chapter.title}")
            if chapter.articles:
                article = chapter.articles[0]
                print(f"   First article: {article.number} - {article.title}")
                print(f"   Content preview: {article.content[:60]}...")

        print("\n✅ Legal parser tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Legal parser tests failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_legal_chunker():
    """Test legal-aware chunker."""
    print("\n" + "=" * 80)
    print("TEST 3: Legal Chunker")
    print("=" * 80)

    from services.data_pipeline.src.chunking import LegalChunker

    from shared.models import Article, Chapter, LegalDocument, Regulation

    chunker = LegalChunker(max_chunk_tokens=256, overlap_sentences=2)

    try:
        # Create sample document
        regulation = Regulation(
            celex_id="test001",
            name="Test Regulation",
            full_title="Test Regulation for Chunking",
        )

        article = Article(
            number="1",
            title="Test Article",
            content="This is a test article. " * 50,  # Long article
            paragraphs=[
                "This is paragraph 1. " * 20,
                "This is paragraph 2. " * 20,
            ],
        )

        chapter = Chapter(number="I", title="Test Chapter", articles=[article])

        document = LegalDocument(regulation=regulation, chapters=[chapter])

        print("\n1. Chunking document...")
        chunks = chunker.chunk_document(document)
        print(f"   Created {len(chunks)} chunks")

        if chunks:
            chunk = chunks[0]
            print("   First chunk metadata:")
            print(f"     - Article: {chunk.metadata.article_number}")
            print(f"     - Chunk: {chunk.metadata.chunk_index}/{chunk.metadata.total_chunks}")
            print(f"     - Domains: {chunk.metadata.domains}")
            print(f"     - Text length: {len(chunk.text)} chars")

        print("\n✅ Legal chunker tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Legal chunker tests failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_embedding_generator():
    """Test embedding generator."""
    print("\n" + "=" * 80)
    print("TEST 4: Embedding Generator")
    print("=" * 80)

    from services.data_pipeline.src.embeddings import EmbeddingGenerator

    from shared.models import Chunk, ChunkMetadata

    try:
        print("\n1. Initializing embedding model (this may take a moment)...")
        generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
            device="cpu",
            batch_size=4,
            show_progress=False,
        )

        info = generator.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Embedding dimension: {info['embedding_dim']}")

        # Create sample chunks
        chunks = [
            Chunk(
                text="High-risk AI systems shall undergo conformity assessment.",
                metadata=ChunkMetadata(
                    regulation_name="AI Act",
                    celex_id="test001",
                    article_number="1",
                ),
            ),
            Chunk(
                text="Data subjects have the right to automated decision-making transparency.",
                metadata=ChunkMetadata(
                    regulation_name="GDPR",
                    celex_id="32016R0679",
                    article_number="22",
                ),
            ),
        ]

        print(f"\n2. Generating embeddings for {len(chunks)} chunks...")
        embedded_chunks = generator.generate_embeddings(chunks, normalize=True)
        print(f"   Generated {len(embedded_chunks)} embeddings")

        if embedded_chunks[0].embedding:
            print(f"   Embedding dimension: {len(embedded_chunks[0].embedding)}")
            print(f"   Sample values: {embedded_chunks[0].embedding[:5]}")

        # Test query embedding
        print("\n3. Generating query embedding...")
        query_emb = generator.generate_query_embedding("What are high-risk AI systems?")
        print(f"   Query embedding dimension: {len(query_emb)}")

        print("\n✅ Embedding generator tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Embedding generator tests failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_qdrant_indexer():
    """Test Qdrant indexer."""
    print("\n" + "=" * 80)
    print("TEST 5: Qdrant Indexer")
    print("=" * 80)

    from services.data_pipeline.src.indexing import QdrantIndexer

    from shared.models import Chunk, ChunkMetadata, RegulationType

    try:
        print("\n1. Connecting to Qdrant...")
        indexer = QdrantIndexer(
            url="http://localhost:6333",
            collection_name="test_collection",
            embedding_dim=384,  # Matches MiniLM model
        )
        print("   ✓ Connected")

        print("\n2. Creating test collection...")
        indexer.create_collection(recreate=True)
        print("   ✓ Collection created")

        # Create sample chunks with fake embeddings
        import random

        chunks = [
            Chunk(
                text=f"This is test chunk {i}",
                metadata=ChunkMetadata(
                    regulation_name="Test Regulation",
                    celex_id="test001",
                    regulation_type=RegulationType.REGULATION,
                    article_number=str(i),
                    chunk_index=0,
                    total_chunks=1,
                ),
                embedding=[random.random() for _ in range(384)],
            )
            for i in range(10)
        ]

        print(f"\n3. Indexing {len(chunks)} chunks...")
        indexed = indexer.index_chunks(chunks, batch_size=5, show_progress=False)
        print(f"   ✓ Indexed {indexed} chunks")

        print("\n4. Getting collection info...")
        info = indexer.get_collection_info()
        print(f"   Vectors: {info['vectors_count']}")
        print(f"   Dimension: {info['vector_size']}")

        print("\n5. Testing search...")
        query_vec = [random.random() for _ in range(384)]
        results = indexer.search(query_vec, limit=3)
        print(f"   ✓ Found {len(results)} results")

        if results:
            print(f"   Top result score: {results[0]['score']:.4f}")

        indexer.close()
        print("\n✅ Qdrant indexer tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Qdrant indexer tests failed: {str(e)}")
        print("   Make sure Qdrant is running: docker-compose up -d qdrant")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ConformAI Data Pipeline Tests")
    print("=" * 80)

    tests = [
        ("EUR-Lex Client", test_eurlex_client),
        ("Legal Parser", test_legal_parser),
        ("Legal Chunker", test_legal_chunker),
        ("Embedding Generator", test_embedding_generator),
        ("Qdrant Indexer", test_qdrant_indexer),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {str(e)}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The data pipeline is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
