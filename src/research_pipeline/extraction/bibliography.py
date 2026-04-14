"""Bibliography extraction from converted Markdown papers.

Parses reference/bibliography sections and extracts structured citation
entries (title, authors, year, identifiers) that can seed the citation
graph without relying solely on the Semantic Scholar API.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Patterns for the start of a bibliography section
_BIB_HEADER_RE = re.compile(
    r"^#{1,3}\s*(References|Bibliography|Works Cited|Literature Cited)\s*$",
    re.IGNORECASE,
)

# arXiv ID patterns: 2401.12345, 2401.12345v2, or old-style hep-ph/0601001
_ARXIV_RE = re.compile(r"(?:arXiv:?\s*)?([\d]{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r"(?:arXiv:?\s*)?([a-z-]+/\d{7}(?:v\d+)?)", re.IGNORECASE)

# DOI pattern
_DOI_RE = re.compile(r"(10\.\d{4,}/[^\s,\]]+)")

# Year in parentheses or after comma: (2024) or , 2024.
_YEAR_RE = re.compile(r"[\(,]\s*((?:19|20)\d{2})\s*[\),.]")


@dataclass
class BibEntry:
    """A single extracted bibliography entry."""

    raw_text: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    arxiv_id: str = ""
    doi: str = ""


def _extract_bib_section(markdown: str) -> str:
    """Extract the bibliography section from markdown text.

    Args:
        markdown: Full markdown content.

    Returns:
        The text of the bibliography section, or empty string if not found.
    """
    lines = markdown.split("\n")
    bib_start = -1

    for i, line in enumerate(lines):
        if _BIB_HEADER_RE.match(line.strip()):
            bib_start = i + 1
            break

    if bib_start < 0:
        return ""

    # Bibliography runs until next heading of same or higher level, or EOF
    bib_lines: list[str] = []
    for line in lines[bib_start:]:
        if re.match(r"^#{1,3}\s+\S", line) and not _BIB_HEADER_RE.match(line.strip()):
            break
        bib_lines.append(line)

    return "\n".join(bib_lines)


def _split_entries(bib_text: str) -> list[str]:
    """Split bibliography text into individual reference entries.

    Handles numbered references like [1], [2] or 1. 2. style.

    Args:
        bib_text: Raw bibliography section text.

    Returns:
        List of individual reference strings.
    """
    # Try numbered references: [1] ... [2] ...
    numbered = re.split(r"\n\s*\[(\d+)\]\s*", bib_text)
    if len(numbered) > 2:
        # numbered[0] is before first [1], then alternating index/text
        entries = []
        for i in range(1, len(numbered), 2):
            if i + 1 < len(numbered):
                entries.append(numbered[i + 1].strip())
        if entries:
            return [e for e in entries if e]

    # Try bullet/dash references
    bullets = re.split(r"\n\s*[-•]\s+", bib_text)
    if len(bullets) > 2:
        return [e.strip() for e in bullets if e.strip()]

    # Fallback: split on blank lines
    paragraphs = re.split(r"\n\s*\n", bib_text)
    return [p.strip() for p in paragraphs if p.strip()]


def _parse_entry(raw: str) -> BibEntry:
    """Parse a single bibliography entry string.

    Args:
        raw: Raw text of one bibliography entry.

    Returns:
        BibEntry with extracted fields.
    """
    entry = BibEntry(raw_text=raw)

    # Extract arXiv ID
    m = _ARXIV_RE.search(raw)
    if m:
        entry.arxiv_id = m.group(1)
    else:
        m = _ARXIV_OLD_RE.search(raw)
        if m:
            entry.arxiv_id = m.group(1)

    # Extract DOI
    m = _DOI_RE.search(raw)
    if m:
        entry.doi = m.group(1).rstrip(".")

    # Extract year
    m = _YEAR_RE.search(raw)
    if m:
        entry.year = int(m.group(1))

    # Extract title: typically the first quoted string or text after authors
    # Try quoted title first: "Title" or *Title*
    title_m = re.search(r'["""]([^"""]+)["""]', raw)
    if not title_m:
        title_m = re.search(r"\*([^*]+)\*", raw)
    if title_m:
        entry.title = title_m.group(1).strip()

    return entry


def extract_bibliography(markdown: str) -> list[BibEntry]:
    """Extract structured bibliography entries from markdown.

    Args:
        markdown: Full markdown content of a converted paper.

    Returns:
        List of BibEntry objects with extracted metadata.
        Empty list if no bibliography section is found.
    """
    bib_text = _extract_bib_section(markdown)
    if not bib_text:
        logger.debug("No bibliography section found")
        return []

    raw_entries = _split_entries(bib_text)
    entries = [_parse_entry(raw) for raw in raw_entries]

    logger.info(
        "Extracted %d bibliography entries (%d with arXiv ID, %d with DOI)",
        len(entries),
        sum(1 for e in entries if e.arxiv_id),
        sum(1 for e in entries if e.doi),
    )
    return entries
