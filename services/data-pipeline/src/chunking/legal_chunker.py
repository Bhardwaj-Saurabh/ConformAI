"""
Legal-Aware Document Chunker

Chunks legal documents while preserving legal meaning and structure.

Strategy:
1. Primary chunks: Full articles (if within token limit)
2. If article too long: Split by paragraph
3. Never split mid-paragraph
4. Preserve article context in metadata
5. Add intelligent overlap between chunks
"""

from datetime import date
from typing import Any, Literal

from langchain.text_splitter import RecursiveCharacterTextSplitter

from shared.models import (
    AIDomain,
    Article,
    Chapter,
    Chunk,
    ChunkMetadata,
    LegalDocument,
    RiskCategory,
)
from shared.utils import get_logger

logger = get_logger(__name__)


class LegalChunker:
    """Legal-aware document chunker."""

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        overlap_sentences: int = 2,
        tokenizer_type: Literal["tiktoken", "transformers", "char"] = "tiktoken",
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize legal chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk
            overlap_sentences: Number of sentences to overlap between chunks
            tokenizer_type: Type of tokenizer ("tiktoken" for OpenAI, "transformers" for HF, "char" for estimation)
            encoding_name: Tiktoken encoding name (cl100k_base for text-embedding-3-*, gpt-4, etc.)
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_sentences = overlap_sentences
        self.tokenizer_type = tokenizer_type
        self.tokenizer = None

        # Load appropriate tokenizer
        if tokenizer_type == "tiktoken":
            # Use tiktoken for OpenAI models
            try:
                import tiktoken

                self.tokenizer = tiktoken.get_encoding(encoding_name)
                logger.info(f"Loaded tiktoken encoding: {encoding_name}")
            except ImportError:
                logger.error("tiktoken not installed. Install with: pip install tiktoken")
                logger.warning("Falling back to character-based estimation")
                self.tokenizer_type = "char"
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
                logger.warning("Falling back to character-based estimation")
                self.tokenizer_type = "char"

        elif tokenizer_type == "transformers":
            # Use HuggingFace tokenizer (for backward compatibility)
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Loaded HuggingFace tokenizer")
            except ImportError:
                logger.error("transformers not installed. Install with: pip install transformers")
                logger.warning("Falling back to character-based estimation")
                self.tokenizer_type = "char"
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace tokenizer: {e}")
                logger.warning("Falling back to character-based estimation")
                self.tokenizer_type = "char"

        # Approximate characters per token (for fallback)
        self.chars_per_token = 4

    def chunk_document(self, document: LegalDocument) -> list[Chunk]:
        """
        Chunk legal document preserving structure.

        Args:
            document: Parsed legal document

        Returns:
            List of chunks with metadata
        """
        logger.info(
            f"Chunking document: {document.regulation.name} ({len(document.chapters)} chapters)"
        )

        all_chunks = []

        for chapter in document.chapters:
            chapter_chunks = self.chunk_chapter(
                chapter=chapter,
                regulation=document.regulation,
            )
            all_chunks.extend(chapter_chunks)

        logger.info(f"Created {len(all_chunks)} chunks from document")
        return all_chunks

    def chunk_chapter(
        self,
        chapter: Chapter,
        regulation: Any,  # Regulation model
    ) -> list[Chunk]:
        """
        Chunk a chapter into semantic units.

        Args:
            chapter: Chapter to chunk
            regulation: Parent regulation metadata

        Returns:
            List of chunks for this chapter
        """
        chunks = []

        for article in chapter.articles:
            article_chunks = self.chunk_article(
                article=article,
                chapter=chapter,
                regulation=regulation,
            )
            chunks.extend(article_chunks)

        return chunks

    def chunk_article(
        self,
        article: Article,
        chapter: Chapter,
        regulation: Any,
    ) -> list[Chunk]:
        """
        Chunk a single article.

        Strategy:
        1. Try to fit entire article in one chunk
        2. If too long, split by paragraph
        3. If paragraph too long, use recursive splitting
        4. Always preserve article context

        Args:
            article: Article to chunk
            chapter: Parent chapter
            regulation: Parent regulation

        Returns:
            List of chunks for this article
        """
        # Calculate tokens for full article
        article_tokens = self._count_tokens(article.content)

        if article_tokens <= self.max_chunk_tokens:
            # Article fits in one chunk
            return [
                self._create_chunk(
                    text=article.content,
                    article=article,
                    chapter=chapter,
                    regulation=regulation,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]

        # Article too long - split by paragraph
        logger.debug(
            f"Article {article.number} exceeds max tokens ({article_tokens} > {self.max_chunk_tokens})"
        )

        if article.paragraphs and len(article.paragraphs) > 1:
            return self._chunk_by_paragraphs(
                article=article,
                chapter=chapter,
                regulation=regulation,
            )
        else:
            # No paragraph structure - use recursive splitting
            return self._chunk_recursively(
                article=article,
                chapter=chapter,
                regulation=regulation,
            )

    def _chunk_by_paragraphs(
        self,
        article: Article,
        chapter: Chapter,
        regulation: Any,
    ) -> list[Chunk]:
        """
        Chunk article by paragraphs, grouping small paragraphs together.

        Args:
            article: Article with paragraphs
            chapter: Parent chapter
            regulation: Parent regulation

        Returns:
            List of paragraph-based chunks
        """
        chunks = []
        current_text = ""
        current_paras = []

        for para_idx, para in enumerate(article.paragraphs):
            para_tokens = self._count_tokens(para)

            # If single paragraph exceeds max, split it recursively
            if para_tokens > self.max_chunk_tokens:
                # Flush current accumulated paragraphs first
                if current_text:
                    chunks.append(
                        self._create_chunk(
                            text=current_text.strip(),
                            article=article,
                            chapter=chapter,
                            regulation=regulation,
                            chunk_index=len(chunks),
                            paragraph_index=current_paras[0] if current_paras else None,
                        )
                    )
                    current_text = ""
                    current_paras = []

                # Split this paragraph
                para_chunks = self._split_long_paragraph(
                    paragraph=para,
                    para_index=para_idx,
                    article=article,
                    chapter=chapter,
                    regulation=regulation,
                    chunk_start_index=len(chunks),
                )
                chunks.extend(para_chunks)
                continue

            # Check if adding this paragraph would exceed max
            combined_text = current_text + "\n\n" + para if current_text else para
            combined_tokens = self._count_tokens(combined_text)

            if combined_tokens > self.max_chunk_tokens and current_text:
                # Flush current chunk
                chunks.append(
                    self._create_chunk(
                        text=current_text.strip(),
                        article=article,
                        chapter=chapter,
                        regulation=regulation,
                        chunk_index=len(chunks),
                        paragraph_index=current_paras[0] if current_paras else None,
                    )
                )
                current_text = para
                current_paras = [para_idx]
            else:
                # Add to current chunk
                current_text = combined_text
                current_paras.append(para_idx)

        # Flush remaining text
        if current_text:
            chunks.append(
                self._create_chunk(
                    text=current_text.strip(),
                    article=article,
                    chapter=chapter,
                    regulation=regulation,
                    chunk_index=len(chunks),
                    paragraph_index=current_paras[0] if current_paras else None,
                )
            )

        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _split_long_paragraph(
        self,
        paragraph: str,
        para_index: int,
        article: Article,
        chapter: Chapter,
        regulation: Any,
        chunk_start_index: int,
    ) -> list[Chunk]:
        """Split a single long paragraph using recursive text splitting."""
        # Use LangChain's RecursiveCharacterTextSplitter
        max_chars = self.max_chunk_tokens * self.chars_per_token

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=100,  # Character overlap
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        splits = splitter.split_text(paragraph)

        chunks = []
        for idx, split in enumerate(splits):
            chunk = self._create_chunk(
                text=split,
                article=article,
                chapter=chapter,
                regulation=regulation,
                chunk_index=chunk_start_index + idx,
                paragraph_index=para_index,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_recursively(
        self,
        article: Article,
        chapter: Chapter,
        regulation: Any,
    ) -> list[Chunk]:
        """Fallback: recursively split article when no paragraph structure."""
        max_chars = self.max_chunk_tokens * self.chars_per_token

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        splits = splitter.split_text(article.content)

        chunks = []
        for idx, split in enumerate(splits):
            chunk = self._create_chunk(
                text=split,
                article=article,
                chapter=chapter,
                regulation=regulation,
                chunk_index=idx,
                total_chunks=len(splits),
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        text: str,
        article: Article,
        chapter: Chapter,
        regulation: Any,
        chunk_index: int,
        total_chunks: int | None = None,
        paragraph_index: int | None = None,
    ) -> Chunk:
        """
        Create chunk with metadata.

        Args:
            text: Chunk text
            article: Source article
            chapter: Source chapter
            regulation: Source regulation
            chunk_index: Index of this chunk within article
            total_chunks: Total chunks for this article (if known)
            paragraph_index: Index of paragraph within article

        Returns:
            Chunk with full metadata
        """
        # Parse effective date
        effective_date = None
        if hasattr(regulation, "effective_date") and regulation.effective_date:
            if isinstance(regulation.effective_date, date):
                effective_date = regulation.effective_date
            elif isinstance(regulation.effective_date, str):
                try:
                    from datetime import datetime

                    effective_date = datetime.fromisoformat(
                        regulation.effective_date
                    ).date()
                except ValueError:
                    pass

        # Classify domains and risk category (heuristic-based)
        domains, risk_category = self._classify_content(text)

        metadata = ChunkMetadata(
            regulation_name=regulation.name,
            celex_id=regulation.celex_id,
            regulation_type=regulation.regulation_type,
            chapter_number=chapter.number,
            chapter_title=chapter.title,
            article_number=article.number,
            article_title=article.title,
            paragraph_index=paragraph_index,
            chunk_index=chunk_index,
            total_chunks=total_chunks or 1,
            effective_date=effective_date,
            version=regulation.version if hasattr(regulation, "version") else "consolidated",
            domains=domains,
            risk_category=risk_category,
            source_url=regulation.url if hasattr(regulation, "url") else None,
        )

        return Chunk(
            text=text,
            metadata=metadata,
            embedding=None,  # Will be added later
        )

    def _classify_content(
        self, text: str
    ) -> tuple[list[AIDomain], RiskCategory | None]:
        """
        Classify chunk content into AI domains and risk categories.

        This is a heuristic-based approach. For production, consider using
        an LLM-based classifier.

        Args:
            text: Chunk text

        Returns:
            Tuple of (domains, risk_category)
        """
        text_lower = text.lower()

        # Domain classification
        domains = []

        domain_keywords = {
            AIDomain.BIOMETRICS: [
                "biometric",
                "facial recognition",
                "fingerprint",
                "iris scan",
                "voice recognition",
            ],
            AIDomain.RECRUITMENT: [
                "recruitment",
                "hiring",
                "employment",
                "job application",
                "candidate",
            ],
            AIDomain.EDUCATION: [
                "education",
                "student",
                "examination",
                "learning",
                "academic",
            ],
            AIDomain.HEALTHCARE: [
                "health",
                "medical",
                "patient",
                "diagnosis",
                "treatment",
            ],
            AIDomain.LAW_ENFORCEMENT: [
                "law enforcement",
                "police",
                "crime",
                "criminal",
                "investigation",
            ],
            AIDomain.CREDIT_SCORING: [
                "credit",
                "creditworthiness",
                "loan",
                "financial",
                "scoring",
            ],
            AIDomain.SOCIAL_SCORING: [
                "social scoring",
                "behavior",
                "trustworthiness",
            ],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                domains.append(domain)

        # Risk category classification
        risk_category = None

        if any(
            keyword in text_lower
            for keyword in [
                "prohibited",
                "shall not",
                "forbidden",
                "manipulation",
                "social scoring",
            ]
        ):
            risk_category = RiskCategory.PROHIBITED

        elif any(
            keyword in text_lower
            for keyword in [
                "high-risk",
                "safety component",
                "biometric identification",
                "critical infrastructure",
            ]
        ):
            risk_category = RiskCategory.HIGH_RISK

        elif any(keyword in text_lower for keyword in ["transparency", "chatbot", "deepfake"]):
            risk_category = RiskCategory.LIMITED_RISK

        if not domains:
            domains = [AIDomain.GENERAL]

        return domains, risk_category

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured tokenizer.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            try:
                if self.tokenizer_type == "tiktoken":
                    # Tiktoken encoding
                    tokens = self.tokenizer.encode(text)
                    return len(tokens)
                elif self.tokenizer_type == "transformers":
                    # HuggingFace tokenizer
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    return len(tokens)
            except Exception as e:
                logger.warning(f"Tokenizer failed: {e}. Using character-based estimation.")

        # Fallback: estimate based on characters
        return len(text) // self.chars_per_token


# Example usage
if __name__ == "__main__":
    from shared.models import Article, Chapter, LegalDocument, Regulation

    # Create sample document
    regulation = Regulation(
        celex_id="32016R0679",
        name="GDPR",
        full_title="General Data Protection Regulation",
    )

    article = Article(
        number="22",
        title="Automated individual decision-making",
        content="""1. The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her.

2. Paragraph 1 shall not apply if the decision:
(a) is necessary for entering into, or performance of, a contract between the data subject and a data controller;
(b) is authorised by Union or Member State law to which the controller is subject and which also lays down suitable measures to safeguard the data subject's rights and freedoms and legitimate interests; or
(c) is based on the data subject's explicit consent.""",
        paragraphs=[
            "The data subject shall have the right not to be subject to a decision based solely on automated processing...",
            "Paragraph 1 shall not apply if...",
        ],
    )

    chapter = Chapter(
        number="III",
        title="Rights of the data subject",
        articles=[article],
    )

    document = LegalDocument(
        regulation=regulation,
        chapters=[chapter],
        raw_text="Sample text",
    )

    # Chunk document
    chunker = LegalChunker(max_chunk_tokens=256)
    chunks = chunker.chunk_document(document)

    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\nChunk {chunk.metadata.chunk_index}/{chunk.metadata.total_chunks}")
        print(f"Article {chunk.metadata.article_number}: {chunk.metadata.article_title}")
        print(f"Domains: {chunk.metadata.domains}")
        print(f"Risk: {chunk.metadata.risk_category}")
        print(f"Text preview: {chunk.text[:100]}...")
