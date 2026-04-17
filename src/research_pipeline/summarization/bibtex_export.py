"""Rich BibTeX export from CandidateRecord objects.

Produces well-formed BibTeX entries with authors, abstracts, DOIs, venues,
and URLs. Works with screened candidates or downloaded papers — does not
require a full synthesis report.

Deep Research Report §B2: BibTeX export for LaTeX workflows.
"""

import json
import logging
import re
from pathlib import Path

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")
_BIBTEX_SPECIAL = re.compile(r"([&%$#_{}~^\\])")


def _escape(value: str) -> str:
    """Escape BibTeX special characters."""
    return _BIBTEX_SPECIAL.sub(r"\\\1", value)


def _make_citation_key(record: CandidateRecord) -> str:
    """Generate a BibTeX citation key from a candidate record.

    Uses first-author surname + year + first word of title.
    Falls back to sanitized arxiv_id if fields are missing.
    """
    parts: list[str] = []
    if record.authors:
        surname = record.authors[0].split()[-1].lower()
        surname = re.sub(r"[^a-z]", "", surname)
        parts.append(surname)

    year = _extract_year(record)
    parts.append(year)

    title_word = re.sub(r"[^a-z]", "", record.title.split()[0].lower()) if record.title else ""
    if title_word:
        parts.append(title_word)

    key = "".join(parts) if parts else record.arxiv_id.replace("/", "_")
    return key


def _extract_year(record: CandidateRecord) -> str:
    """Extract publication year from record fields."""
    if record.year:
        return str(record.year)
    match = _ARXIV_ID_RE.match(record.arxiv_id)
    if match:
        yy = int(match.group(1))
        century = 20 if yy < 90 else 19
        return str(century * 100 + yy)
    if record.published:
        return str(record.published.year)
    return "unknown"


def _format_authors(authors: list[str]) -> str:
    """Format author list for BibTeX (joined with ' and ')."""
    return " and ".join(_escape(a) for a in authors)


def candidate_to_bibtex(record: CandidateRecord) -> str:
    """Convert a CandidateRecord to a BibTeX entry string.

    Uses @article for arXiv papers, @misc otherwise.

    Args:
        record: The candidate record to convert.

    Returns:
        Formatted BibTeX entry string.
    """
    is_arxiv = _ARXIV_ID_RE.match(record.arxiv_id) is not None
    entry_type = "article" if is_arxiv else "misc"
    key = _make_citation_key(record)
    year = _extract_year(record)

    lines = [f"@{entry_type}{{{key},"]
    lines.append(f"  title = {{{_escape(record.title)}}},")

    if record.authors:
        lines.append(f"  author = {{{_format_authors(record.authors)}}},")

    lines.append(f"  year = {{{year}}},")

    if is_arxiv:
        lines.append(f"  eprint = {{{record.arxiv_id}}},")
        lines.append("  archivePrefix = {arXiv},")
        if record.primary_category:
            lines.append(
                f"  primaryClass = {{{_escape(record.primary_category)}}},"
            )

    if record.doi:
        lines.append(f"  doi = {{{record.doi}}},")

    if record.venue:
        lines.append(f"  journal = {{{_escape(record.venue)}}},")

    if record.abs_url:
        lines.append(f"  url = {{{record.abs_url}}},")

    if record.abstract:
        abstract_clean = record.abstract.replace("\n", " ").strip()
        if abstract_clean:
            lines.append(f"  abstract = {{{_escape(abstract_clean)}}},")

    lines.append("}")
    return "\n".join(lines)


def export_candidates_bibtex(
    candidates: list[CandidateRecord],
    output_path: Path,
) -> int:
    """Export a list of candidates as a BibTeX file.

    Args:
        candidates: List of candidate records to export.
        output_path: Destination .bib file path.

    Returns:
        Number of entries written.
    """
    entries = [candidate_to_bibtex(c) for c in candidates]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    logger.info("Exported %d BibTeX entries to %s", len(entries), output_path)
    return len(entries)


def load_candidates_from_jsonl(path: Path) -> list[CandidateRecord]:
    """Load CandidateRecord objects from a JSONL file.

    Args:
        path: Path to a candidates JSONL file.

    Returns:
        List of parsed CandidateRecord objects.
    """
    records: list[CandidateRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(CandidateRecord.model_validate(data))
    return records
