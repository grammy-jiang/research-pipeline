"""Atom XML parser: arXiv API response → CandidateRecord list."""

import logging
import re
from datetime import UTC, datetime

from lxml import etree  # type: ignore[import-untyped]

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"

_ID_PATTERN = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(v\d+)?")

# Secure XML parser: disable network access and entity resolution
_SAFE_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
)


def _parse_arxiv_id(id_text: str) -> tuple[str, str]:
    """Extract base arXiv ID and version from an Atom entry ID.

    Args:
        id_text: The ``<id>`` field from an Atom entry.

    Returns:
        Tuple of ``(base_id, version)``.
    """
    match = _ID_PATTERN.search(id_text)
    if match:
        base_id = match.group(1)
        version = match.group(2) or "v1"
        return base_id, version
    # Fallback: try to extract from the URL directly
    parts = id_text.rstrip("/").split("/")
    raw = parts[-1] if parts else id_text
    if "v" in raw:
        idx = raw.rfind("v")
        return raw[:idx], raw[idx:]
    return raw, "v1"


def _parse_datetime(text: str) -> datetime:
    """Parse an ISO 8601 datetime string to timezone-aware UTC datetime."""
    text = text.strip()
    # Handle the Z suffix
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _find_text(element: etree._Element, tag: str, ns: str = ATOM_NS) -> str:
    """Find text content of a child element, returning empty string if missing."""
    child = element.find(f"{{{ns}}}{tag}")
    if child is not None and child.text:
        return child.text.strip()  # type: ignore[no-any-return]
    return ""


def parse_atom_response(xml_text: str) -> list[CandidateRecord]:
    """Parse an arXiv Atom XML response into CandidateRecord objects.

    Args:
        xml_text: Raw XML response text from the arXiv API.

    Returns:
        List of parsed candidate records.

    Raises:
        etree.XMLSyntaxError: If the XML is malformed.
    """
    root = etree.fromstring(xml_text.encode(), parser=_SAFE_PARSER)
    entries = root.findall(f"{{{ATOM_NS}}}entry")
    logger.info("Parsing %d entries from Atom response", len(entries))

    candidates: list[CandidateRecord] = []
    for entry in entries:
        try:
            candidate = _parse_entry(entry)
            candidates.append(candidate)
        except Exception as exc:
            entry_id = _find_text(entry, "id")
            logger.error("Failed to parse entry %s: %s", entry_id, exc)

    logger.info("Successfully parsed %d candidates", len(candidates))
    return candidates


def _parse_entry(entry: etree._Element) -> CandidateRecord:
    """Parse a single Atom entry into a CandidateRecord."""
    raw_id = _find_text(entry, "id")
    arxiv_id, version = _parse_arxiv_id(raw_id)

    title = _find_text(entry, "title")
    # Normalize whitespace in title
    title = " ".join(title.split())

    abstract = _find_text(entry, "summary")
    abstract = abstract.strip()

    published = _parse_datetime(_find_text(entry, "published"))
    updated = _parse_datetime(_find_text(entry, "updated"))

    # Authors
    authors: list[str] = []
    for author_el in entry.findall(f"{{{ATOM_NS}}}author"):
        name = _find_text(author_el, "name")
        if name:
            authors.append(name)

    # Categories
    categories: list[str] = []
    for cat_el in entry.findall(f"{{{ATOM_NS}}}category"):
        term = cat_el.get("term", "")
        if term:
            categories.append(term)

    # Primary category from arxiv namespace
    primary_cat_el = entry.find(f"{{{ARXIV_NS}}}primary_category")
    primary_category = ""
    if primary_cat_el is not None:
        primary_category = primary_cat_el.get("term", "")
    if not primary_category and categories:
        primary_category = categories[0]

    # Links
    abs_url = f"https://arxiv.org/abs/{arxiv_id}{version}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}{version}"

    for link_el in entry.findall(f"{{{ATOM_NS}}}link"):
        if link_el.get("title") == "pdf":
            pdf_url = link_el.get("href", pdf_url)
        elif link_el.get("rel") == "alternate":
            abs_url = link_el.get("href", abs_url)

    return CandidateRecord(
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        authors=authors,
        published=published,
        updated=updated,
        categories=categories,
        primary_category=primary_category,
        abstract=abstract,
        abs_url=abs_url,
        pdf_url=pdf_url,
    )


def parse_total_results(xml_text: str) -> int:
    """Extract the totalResults count from an arXiv Atom response.

    Args:
        xml_text: Raw XML response text.

    Returns:
        Total number of results reported by arXiv.
    """
    root = etree.fromstring(xml_text.encode(), parser=_SAFE_PARSER)
    total_el = root.find(f"{{{OPENSEARCH_NS}}}totalResults")
    if total_el is not None and total_el.text:
        return int(total_el.text)
    return 0
