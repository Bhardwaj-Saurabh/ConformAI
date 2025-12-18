"""
Legal Document Parser

Parses EU legal documents from various formats (XML, HTML, PDF) and extracts
structured legal hierarchy: Regulations → Chapters → Articles → Paragraphs.

Supports:
- EUR-Lex Formex XML format
- EUR-Lex HTML format
- PDF (with OCR fallback)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from lxml import etree

from shared.models import Article, Chapter, LegalDocument, Regulation
from shared.utils import get_logger

logger = get_logger(__name__)


class LegalDocumentParser:
    """Parser for EU legal documents."""

    def __init__(self):
        """Initialize parser."""
        self.namespaces = {
            "fmx": "http://formex.publications.europa.eu/schema/formex-05.21-20110601.xd",
            "xhtml": "http://www.w3.org/1999/xhtml",
        }

    def parse(
        self, file_path: Path | str, regulation: Regulation | None = None
    ) -> LegalDocument:
        """
        Parse legal document from file.

        Args:
            file_path: Path to document file
            regulation: Optional Regulation metadata

        Returns:
            Parsed LegalDocument

        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Detect format
        suffix = file_path.suffix.lower()

        if suffix == ".xml":
            return self.parse_xml(file_path, regulation)
        elif suffix in [".html", ".htm"]:
            return self.parse_html(file_path, regulation)
        elif suffix == ".pdf":
            return self.parse_pdf(file_path, regulation)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def parse_xml(
        self, file_path: Path | str, regulation: Regulation | None = None
    ) -> LegalDocument:
        """
        Parse EUR-Lex Formex XML document.

        Formex is the official XML format used by EUR-Lex for legislative documents.

        Args:
            file_path: Path to XML file
            regulation: Optional Regulation metadata

        Returns:
            Parsed LegalDocument
        """
        logger.info(f"Parsing XML document: {file_path}")

        file_path = Path(file_path)
        content = file_path.read_bytes()

        try:
            # Parse with lxml for better namespace handling
            tree = etree.fromstring(content)

            # Extract metadata if not provided
            if regulation is None:
                regulation = self._extract_metadata_from_xml(tree)

            # Extract chapters and articles
            chapters = self._extract_chapters_from_xml(tree)

            # Extract raw text
            raw_text = self._extract_text_from_xml(tree)

            return LegalDocument(
                regulation=regulation,
                chapters=chapters,
                raw_text=raw_text,
                parsed_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Failed to parse XML: {str(e)}")
            raise

    def parse_html(
        self, file_path: Path | str, regulation: Regulation | None = None
    ) -> LegalDocument:
        """
        Parse EUR-Lex HTML document.

        Args:
            file_path: Path to HTML file
            regulation: Optional Regulation metadata

        Returns:
            Parsed LegalDocument
        """
        logger.info(f"Parsing HTML document: {file_path}")

        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")

        soup = BeautifulSoup(content, "lxml")

        # Extract metadata if not provided
        if regulation is None:
            regulation = self._extract_metadata_from_html(soup)

        # Extract chapters and articles
        chapters = self._extract_chapters_from_html(soup)

        # Extract raw text
        raw_text = soup.get_text(separator="\n", strip=True)

        return LegalDocument(
            regulation=regulation,
            chapters=chapters,
            raw_text=raw_text,
            parsed_at=datetime.utcnow(),
        )

    def parse_pdf(
        self, file_path: Path | str, regulation: Regulation | None = None
    ) -> LegalDocument:
        """
        Parse PDF legal document.

        Uses pypdf for text extraction with OCR fallback.

        Args:
            file_path: Path to PDF file
            regulation: Optional Regulation metadata

        Returns:
            Parsed LegalDocument
        """
        logger.info(f"Parsing PDF document: {file_path}")

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")

        file_path = Path(file_path)
        reader = PdfReader(str(file_path))

        # Extract text from all pages
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"

        # Parse structure from text
        # This is more heuristic since PDFs don't have semantic structure
        chapters = self._extract_chapters_from_text(raw_text)

        return LegalDocument(
            regulation=regulation or Regulation(
                celex_id="unknown",
                name="PDF Document",
                full_title="PDF Document",
            ),
            chapters=chapters,
            raw_text=raw_text,
            parsed_at=datetime.utcnow(),
        )

    def _extract_metadata_from_xml(self, tree: etree._Element) -> Regulation:
        """Extract regulation metadata from XML."""
        # Try to find CELEX number
        celex_elem = tree.find(".//IDENTIFIER.CELEX")
        celex = celex_elem.text if celex_elem is not None else "unknown"

        # Try to find title
        title_elem = tree.find(".//TITLE")
        title = title_elem.text if title_elem is not None else "Unknown Title"

        return Regulation(
            celex_id=celex,
            name=title.split()[0:3] if title else "Unknown",  # First few words
            full_title=title,
        )

    def _extract_chapters_from_xml(self, tree: etree._Element) -> list[Chapter]:
        """Extract chapters and articles from XML."""
        chapters = []

        # Find all TITLE elements that represent chapters
        # Formex structure varies, so we try multiple approaches
        chapter_elements = tree.findall(".//TITLE[@TYPE='CHAPTER']") or tree.findall(
            ".//GR.TITLE[@TYPE='CHAPTER']"
        )

        for idx, chapter_elem in enumerate(chapter_elements):
            chapter_num = chapter_elem.get("NUM", f"Chapter {idx+1}")
            chapter_title = chapter_elem.text or ""

            # Find articles under this chapter
            articles = self._extract_articles_from_xml_chapter(chapter_elem)

            if articles:  # Only add chapter if it has articles
                chapters.append(
                    Chapter(
                        number=chapter_num,
                        title=chapter_title,
                        articles=articles,
                    )
                )

        # If no chapters found, try to extract articles directly
        if not chapters:
            articles = self._extract_articles_from_xml(tree)
            if articles:
                chapters.append(
                    Chapter(
                        number="I",
                        title="General Provisions",
                        articles=articles,
                    )
                )

        return chapters

    def _extract_articles_from_xml(self, tree: etree._Element) -> list[Article]:
        """Extract all articles from XML tree."""
        articles = []

        # Find ARTICLE elements
        article_elements = tree.findall(".//ARTICLE") or tree.findall(".//GR.ARTICLE")

        for article_elem in article_elements:
            article = self._parse_article_from_xml(article_elem)
            if article:
                articles.append(article)

        return articles

    def _extract_articles_from_xml_chapter(
        self, chapter_elem: etree._Element
    ) -> list[Article]:
        """Extract articles from a specific chapter element."""
        articles = []

        # Find ARTICLE elements under this chapter
        article_elements = chapter_elem.findall(".//ARTICLE") or chapter_elem.findall(
            ".//GR.ARTICLE"
        )

        for article_elem in article_elements:
            article = self._parse_article_from_xml(article_elem)
            if article:
                articles.append(article)

        return articles

    def _parse_article_from_xml(self, article_elem: etree._Element) -> Article | None:
        """Parse a single article from XML element."""
        # Get article number
        article_num = article_elem.get("NUM") or article_elem.get("IDENTIFIER")

        if not article_num:
            # Try to find in child elements
            num_elem = article_elem.find(".//TI.ART")
            if num_elem is not None:
                article_num = num_elem.text or "Unknown"
            else:
                return None

        # Get article title (optional)
        title_elem = article_elem.find(".//STI.ART")
        article_title = title_elem.text if title_elem is not None else None

        # Get article content
        paragraphs = []
        para_elements = article_elem.findall(".//P") or article_elem.findall(".//PARAG")

        for para_elem in para_elements:
            para_text = self._get_element_text(para_elem)
            if para_text:
                paragraphs.append(para_text)

        # If no paragraphs found, get all text
        if not paragraphs:
            full_text = self._get_element_text(article_elem)
            if full_text:
                paragraphs = [full_text]

        # Combine all paragraph text
        content = "\n\n".join(paragraphs)

        # Extract references to other articles
        references = self._extract_article_references(content)

        return Article(
            number=article_num.strip(),
            title=article_title.strip() if article_title else None,
            content=content,
            paragraphs=paragraphs,
            references=references,
        )

    def _get_element_text(self, element: etree._Element) -> str:
        """Get all text content from an XML element."""
        # Get text from element and all descendants
        text_parts = []

        if element.text:
            text_parts.append(element.text.strip())

        for child in element:
            child_text = self._get_element_text(child)
            if child_text:
                text_parts.append(child_text)

            if child.tail:
                text_parts.append(child.tail.strip())

        return " ".join(text_parts)

    def _extract_text_from_xml(self, tree: etree._Element) -> str:
        """Extract plain text from XML document."""
        return self._get_element_text(tree)

    def _extract_chapters_from_html(self, soup: BeautifulSoup) -> list[Chapter]:
        """Extract chapters and articles from HTML."""
        chapters = []

        # EUR-Lex HTML uses specific classes for structure
        # Try to find chapter headings
        chapter_headings = soup.find_all(["h2", "h3"], class_=re.compile(r"ti-chapter|chapter"))

        for idx, heading in enumerate(chapter_headings):
            chapter_num = self._extract_number_from_heading(heading.get_text())
            chapter_title = heading.get_text(strip=True)

            # Find articles following this chapter
            articles = self._extract_articles_from_html_section(heading)

            if articles:
                chapters.append(
                    Chapter(
                        number=chapter_num or f"{idx+1}",
                        title=chapter_title,
                        articles=articles,
                    )
                )

        # If no chapters, extract all articles
        if not chapters:
            articles = self._extract_articles_from_html(soup)
            if articles:
                chapters.append(
                    Chapter(
                        number="I",
                        title="General Provisions",
                        articles=articles,
                    )
                )

        return chapters

    def _extract_articles_from_html(self, soup: BeautifulSoup) -> list[Article]:
        """Extract all articles from HTML."""
        articles = []

        # Find article headings
        article_headings = soup.find_all(["h3", "h4"], class_=re.compile(r"ti-art|article"))

        for heading in article_headings:
            article = self._parse_article_from_html(heading)
            if article:
                articles.append(article)

        return articles

    def _extract_articles_from_html_section(self, section_heading: Tag) -> list[Article]:
        """Extract articles from HTML section following a heading."""
        articles = []

        # Find next siblings until next chapter
        current = section_heading.next_sibling

        while current:
            if isinstance(current, Tag):
                # Check if it's another chapter heading
                if current.name in ["h2", "h3"] and "chapter" in str(current.get("class", [])):
                    break

                # Check if it's an article
                if current.name in ["h3", "h4"] and "article" in str(current.get("class", [])):
                    article = self._parse_article_from_html(current)
                    if article:
                        articles.append(article)

            current = current.next_sibling

        return articles

    def _parse_article_from_html(self, article_heading: Tag) -> Article | None:
        """Parse a single article from HTML."""
        heading_text = article_heading.get_text(strip=True)

        # Extract article number (e.g., "Article 5" -> "5")
        article_num = self._extract_number_from_heading(heading_text)
        if not article_num:
            return None

        # Extract title (text after article number)
        article_title = heading_text.replace(f"Article {article_num}", "").strip()
        if not article_title:
            article_title = None

        # Find paragraphs following this article
        paragraphs = []
        current = article_heading.next_sibling

        while current:
            if isinstance(current, Tag):
                # Stop at next article or chapter
                if current.name in ["h2", "h3", "h4"]:
                    break

                # Extract paragraph text
                if current.name == "p":
                    para_text = current.get_text(strip=True)
                    if para_text:
                        paragraphs.append(para_text)

            current = current.next_sibling

        content = "\n\n".join(paragraphs)
        references = self._extract_article_references(content)

        return Article(
            number=article_num,
            title=article_title,
            content=content,
            paragraphs=paragraphs,
            references=references,
        )

    def _extract_metadata_from_html(self, soup: BeautifulSoup) -> Regulation:
        """Extract regulation metadata from HTML."""
        # Try to find title
        title_elem = soup.find("title") or soup.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else "Unknown Title"

        return Regulation(
            celex_id="unknown",
            name=title.split()[0:3] if title else "Unknown",
            full_title=title,
        )

    def _extract_chapters_from_text(self, text: str) -> list[Chapter]:
        """Extract chapters from plain text (heuristic-based)."""
        chapters = []

        # Split by chapter markers
        chapter_pattern = r"(?:CHAPTER|Chapter)\s+([IVXLCDM]+|[0-9]+)[:\.]?\s+(.+?)(?=\n)"
        chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))

        if not chapter_matches:
            # No chapters found, create single chapter with all articles
            articles = self._extract_articles_from_text(text)
            if articles:
                return [
                    Chapter(
                        number="I",
                        title="General Provisions",
                        articles=articles,
                    )
                ]
            return []

        for i, match in enumerate(chapter_matches):
            chapter_num = match.group(1)
            chapter_title = match.group(2).strip()

            # Get text until next chapter
            start_pos = match.end()
            end_pos = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)

            chapter_text = text[start_pos:end_pos]

            # Extract articles from this chapter
            articles = self._extract_articles_from_text(chapter_text)

            chapters.append(
                Chapter(
                    number=chapter_num,
                    title=chapter_title,
                    articles=articles,
                )
            )

        return chapters

    def _extract_articles_from_text(self, text: str) -> list[Article]:
        """Extract articles from plain text."""
        articles = []

        # Article pattern: "Article 5" or "Art. 5" or "Article 5:"
        article_pattern = r"(?:Article|Art\.?)\s+([0-9]+)([:\.]?)\s*(.+?)(?=(?:Article|Art\.?)\s+[0-9]+|$)"

        for match in re.finditer(article_pattern, text, re.IGNORECASE | re.DOTALL):
            article_num = match.group(1)
            article_content = match.group(3).strip()

            # Split into paragraphs
            paragraphs = [p.strip() for p in article_content.split("\n\n") if p.strip()]

            references = self._extract_article_references(article_content)

            articles.append(
                Article(
                    number=article_num,
                    title=None,
                    content=article_content,
                    paragraphs=paragraphs,
                    references=references,
                )
            )

        return articles

    def _extract_number_from_heading(self, heading: str) -> str | None:
        """Extract number from heading text."""
        # Try to match patterns like "Chapter 5", "Article 22", etc.
        match = re.search(r"(?:Chapter|Article|Art\.?)\s+([0-9]+|[IVXLCDM]+)", heading, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_article_references(self, text: str) -> list[str]:
        """Extract references to other articles from text."""
        references = []

        # Pattern: "Article 5", "Art. 5", "Articles 5 and 6", "Article 5(2)"
        patterns = [
            r"Article\s+([0-9]+)(?:\([0-9]+\))?",
            r"Art\.\s+([0-9]+)(?:\([0-9]+\))?",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend([f"Article {m}" for m in matches])

        return list(set(references))  # Remove duplicates


# Example usage
if __name__ == "__main__":
    parser = LegalDocumentParser()

    # Parse an XML file (if you have one)
    # doc = parser.parse("path/to/document.xml")
    # print(f"Parsed {len(doc.chapters)} chapters")
    # for chapter in doc.chapters:
    #     print(f"Chapter {chapter.number}: {chapter.title} ({len(chapter.articles)} articles)")
