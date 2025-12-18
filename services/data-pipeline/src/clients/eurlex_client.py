"""
EUR-Lex API Client

Client for fetching EU legal documents from EUR-Lex SPARQL endpoint and REST API.

Official EUR-Lex endpoints:
- SPARQL: https://publications.europa.eu/webapi/rdf/sparql
- REST API: https://eur-lex.europa.eu/legal-content/
- Search API: https://eur-lex.europa.eu/search.html
"""

import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from shared.models import Regulation, RegulationType
from shared.utils import get_logger

logger = get_logger(__name__)


class EURLexClient:
    """Client for EUR-Lex API."""

    def __init__(
        self,
        sparql_endpoint: str = "https://publications.europa.eu/webapi/rdf/sparql",
        rest_base_url: str = "https://eur-lex.europa.eu",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """
        Initialize EUR-Lex client.

        Args:
            sparql_endpoint: SPARQL endpoint URL
            rest_base_url: REST API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.sparql_endpoint = sparql_endpoint
        self.rest_base_url = rest_base_url
        self.timeout = timeout
        self.max_retries = max_retries

        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "ConformAI/0.1.0 (EU AI Compliance Research Bot)",
                "Accept": "application/sparql-results+json, application/xml, text/html",
            },
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _execute_sparql(self, query: str) -> dict[str, Any]:
        """
        Execute SPARQL query against EUR-Lex endpoint.

        Args:
            query: SPARQL query string

        Returns:
            Query results as dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        logger.debug(f"Executing SPARQL query: {query[:200]}...")

        response = self.client.post(
            self.sparql_endpoint,
            headers={"Accept": "application/sparql-results+json"},
            data={"query": query},
        )
        response.raise_for_status()

        return response.json()

    def search_recent_documents(
        self,
        start_date: date | None = None,
        limit: int = 50,
        keywords: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for recent EU legal documents.

        Args:
            start_date: Filter documents from this date onwards
            limit: Maximum number of results
            keywords: Keywords to search for in document titles

        Returns:
            List of document metadata dictionaries
        """
        if start_date is None:
            start_date = date(2024, 1, 1)

        # Build keyword filter
        keyword_filter = ""
        if keywords:
            keyword_patterns = " || ".join(
                [f'CONTAINS(LCASE(str(?title)), "{kw.lower()}")' for kw in keywords]
            )
            keyword_filter = f"FILTER ({keyword_patterns})"

        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT DISTINCT ?work ?celex ?title ?date ?type
        WHERE {{
            ?work cdm:work_has_resource-type ?typeUri .
            ?work cdm:resource_legal_id_celex ?celex .
            ?work cdm:work_date_document ?date .

            OPTIONAL {{ ?work cdm:work_title ?titleObj . }}
            OPTIONAL {{ ?titleObj skos:prefLabel ?title . }}

            FILTER (?date >= "{start_date.isoformat()}"^^xsd:date)
            {keyword_filter}

            BIND(REPLACE(STR(?typeUri), ".*resource-type/", "") AS ?type)
        }}
        ORDER BY DESC(?date)
        LIMIT {limit}
        """

        try:
            results = self._execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            documents = []
            for item in bindings:
                doc = {
                    "work_uri": item.get("work", {}).get("value"),
                    "celex": item.get("celex", {}).get("value"),
                    "title": item.get("title", {}).get("value", "Unknown Title"),
                    "date": item.get("date", {}).get("value"),
                    "type": item.get("type", {}).get("value", "unknown"),
                }
                documents.append(doc)

            logger.info(f"Found {len(documents)} documents from EUR-Lex")
            return documents

        except Exception as e:
            logger.error(f"Failed to search EUR-Lex: {str(e)}")
            raise

    def search_ai_related_documents(
        self, start_date: date | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Search for AI-related EU documents.

        Searches for documents containing keywords related to AI, data protection, GDPR.

        Args:
            start_date: Filter documents from this date onwards
            limit: Maximum number of results

        Returns:
            List of AI-related document metadata
        """
        keywords = [
            "artificial intelligence",
            "AI system",
            "data protection",
            "GDPR",
            "automated decision",
            "algorithmic",
            "machine learning",
        ]

        return self.search_recent_documents(
            start_date=start_date, limit=limit, keywords=keywords
        )

    def get_document_by_celex(self, celex: str) -> dict[str, Any]:
        """
        Get detailed document metadata by CELEX number.

        Args:
            celex: CELEX identifier (e.g., '32016R0679' for GDPR)

        Returns:
            Document metadata dictionary
        """
        query = f"""
        PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT DISTINCT ?work ?title ?date ?type
        WHERE {{
            ?work cdm:resource_legal_id_celex "{celex}" .
            ?work cdm:work_has_resource-type ?typeUri .
            ?work cdm:work_date_document ?date .

            OPTIONAL {{ ?work cdm:work_title ?titleObj . }}
            OPTIONAL {{ ?titleObj skos:prefLabel ?title . }}

            BIND(REPLACE(STR(?typeUri), ".*resource-type/", "") AS ?type)
        }}
        LIMIT 1
        """

        try:
            results = self._execute_sparql(query)
            bindings = results.get("results", {}).get("bindings", [])

            if not bindings:
                raise ValueError(f"Document with CELEX {celex} not found")

            item = bindings[0]
            return {
                "work_uri": item.get("work", {}).get("value"),
                "celex": celex,
                "title": item.get("title", {}).get("value", "Unknown Title"),
                "date": item.get("date", {}).get("value"),
                "type": item.get("type", {}).get("value", "unknown"),
            }

        except Exception as e:
            logger.error(f"Failed to get document {celex}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def download_document(
        self, celex: str, format: str = "xml", language: str = "EN"
    ) -> bytes:
        """
        Download document content from EUR-Lex.

        Args:
            celex: CELEX identifier
            format: Document format ('xml', 'pdf', 'html')
            language: Document language code (default: 'EN')

        Returns:
            Document content as bytes

        Raises:
            httpx.HTTPError: If download fails
        """
        format_map = {
            "xml": "fmx",  # Formex XML
            "pdf": "pdf",
            "html": "html",
        }

        format_code = format_map.get(format.lower(), "fmx")

        # EUR-Lex document URL structure:
        # https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679&format=XML
        url = f"{self.rest_base_url}/legal-content/{language}/TXT/"

        params = {"uri": f"CELEX:{celex}"}

        if format_code != "html":
            params["format"] = format_code.upper()

        logger.info(f"Downloading {celex} in {format} format...")

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()

            # Add a small delay to be polite to EUR-Lex servers
            time.sleep(0.5)

            logger.info(f"Successfully downloaded {celex} ({len(response.content)} bytes)")
            return response.content

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to download {celex}: HTTP {e.response.status_code}")
            raise

    def download_document_to_file(
        self,
        celex: str,
        output_path: Path | str,
        format: str = "xml",
        language: str = "EN",
    ) -> Path:
        """
        Download document and save to file.

        Args:
            celex: CELEX identifier
            output_path: Path to save the document
            format: Document format ('xml', 'pdf', 'html')
            language: Document language code

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.download_document(celex, format=format, language=language)

        output_path.write_bytes(content)
        logger.info(f"Saved {celex} to {output_path}")

        return output_path

    def get_gdpr_document(self) -> dict[str, Any]:
        """
        Get GDPR (General Data Protection Regulation) document metadata.

        CELEX: 32016R0679

        Returns:
            GDPR document metadata
        """
        return self.get_document_by_celex("32016R0679")

    def get_ai_act_document(self) -> dict[str, Any]:
        """
        Get EU AI Act document metadata.

        Note: The final AI Act CELEX number will be different.
        This is a placeholder - update with actual CELEX once published.

        Returns:
            AI Act document metadata
        """
        # TODO: Update with actual AI Act CELEX when available
        # For now, search for AI Act related documents
        logger.warning("AI Act CELEX not yet finalized. Searching for related documents...")

        docs = self.search_ai_related_documents(start_date=date(2021, 1, 1), limit=10)

        # Filter for AI Act proposals
        ai_act_docs = [
            doc for doc in docs if "artificial intelligence" in doc["title"].lower()
        ]

        if ai_act_docs:
            return ai_act_docs[0]
        else:
            raise ValueError("AI Act document not found")

    def extract_celex_metadata(self, celex: str) -> Regulation:
        """
        Extract metadata from CELEX number and fetch document details.

        Args:
            celex: CELEX identifier

        Returns:
            Regulation model with metadata
        """
        doc = self.get_document_by_celex(celex)

        # Parse CELEX structure:
        # Format: CYYYXTNNNN
        # C: Sector (3 = EU law)
        # YYYY: Year
        # X: Type (R = Regulation, L = Directive, D = Decision)
        # NNNN: Sequential number

        year = celex[1:5] if len(celex) >= 5 else None
        type_code = celex[5] if len(celex) > 5 else None

        type_map = {
            "R": RegulationType.REGULATION,
            "L": RegulationType.DIRECTIVE,
            "D": RegulationType.DECISION,
        }

        reg_type = type_map.get(type_code, RegulationType.REGULATION)

        # Parse date
        doc_date = None
        if doc.get("date"):
            try:
                doc_date = datetime.fromisoformat(doc["date"].replace("Z", "+00:00")).date()
            except ValueError:
                pass

        return Regulation(
            celex_id=celex,
            name=self._extract_regulation_name(doc["title"]),
            full_title=doc["title"],
            regulation_type=reg_type,
            adoption_date=doc_date,
            effective_date=doc_date,  # Simplified - may need separate query
            url=f"{self.rest_base_url}/legal-content/EN/TXT/?uri=CELEX:{celex}",
            version="consolidated",
            is_active=True,
        )

    def _extract_regulation_name(self, title: str) -> str:
        """
        Extract short regulation name from full title.

        Args:
            title: Full regulation title

        Returns:
            Short name (e.g., 'GDPR', 'AI Act')
        """
        title_lower = title.lower()

        # Known regulation mappings
        if "general data protection" in title_lower or "2016/679" in title:
            return "GDPR"
        elif "artificial intelligence" in title_lower and "regulation" in title_lower:
            return "AI Act"
        elif "edpb" in title_lower:
            return "EDPB Guidelines"
        else:
            # Extract first meaningful words
            words = title.split()[:3]
            return " ".join(words)

    def close(self):
        """Close HTTP client."""
        self.client.close()


# Example usage
if __name__ == "__main__":
    client = EURLexClient()

    # Search for AI-related documents
    docs = client.search_ai_related_documents(start_date=date(2023, 1, 1), limit=10)
    for doc in docs:
        print(f"CELEX: {doc['celex']}")
        print(f"Title: {doc['title']}")
        print(f"Date: {doc['date']}")
        print("-" * 80)

    # Download GDPR
    gdpr_content = client.download_document("32016R0679", format="xml")
    print(f"Downloaded GDPR: {len(gdpr_content)} bytes")

    client.close()
