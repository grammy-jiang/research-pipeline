"""FACT citation verification scoring.

Verifies that citations in research reports refer to real papers in the
corpus. Computes Citation Accuracy and Effective Citation Ratio metrics
as defined by the FACT framework (DeepResearch Bench).
"""

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class FACTScore(BaseModel):
    """Citation verification result from the FACT framework.

    Captures how faithfully a research report cites the underlying paper
    corpus.  ``citation_accuracy`` measures the fraction of in-text
    citations that resolve to a real paper; ``effective_citation_ratio``
    measures how much of the corpus is actually referenced.
    """

    citation_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of citations that resolve to a real paper (0-1).",
    )
    effective_citation_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of corpus papers cited at least once (0-1).",
    )
    total_citations: int = Field(
        ge=0,
        description="Total number of citation references found in the text.",
    )
    verified_citations: int = Field(
        ge=0,
        description="Number of citations that matched a known paper.",
    )
    unsupported_citations: list[str] = Field(
        default_factory=list,
        description="Citation refs that could not be matched to any paper.",
    )
    uncited_papers: list[str] = Field(
        default_factory=list,
        description="Paper IDs from the corpus that were never cited.",
    )
    citation_map: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of citation ref → matched paper_id.",
    )


# ---------------------------------------------------------------------------
# Citation extraction patterns
# ---------------------------------------------------------------------------

# Matches [n] where n is one or more digits — e.g. [1], [12]
_NUMERIC_CITE_RE = re.compile(r"\[(\d+)\]")

# Matches arXiv-style ids — e.g. [2301.12345], [hep-th/9901001]
_ARXIV_CITE_RE = re.compile(r"\[(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+/\d{7}(?:v\d+)?)\]")

# Matches author-year citations — e.g. [Author et al., 2024], [Smith, 2023]
_AUTHOR_YEAR_CITE_RE = re.compile(
    r"\[([A-Z][A-Za-z'-]+(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'-]+))?,"
    r"\s*\d{4}[a-z]?)\]"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation for fuzzy title comparison.

    Args:
        title: Raw paper title.

    Returns:
        Normalized title string.
    """
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _extract_citations(text: str) -> list[tuple[str, str]]:
    """Extract all citation references from markdown text.

    Each returned tuple is ``(raw_ref, kind)`` where *kind* is one of
    ``"numeric"``, ``"arxiv"``, or ``"author_year"``.

    Args:
        text: Markdown report content.

    Returns:
        List of ``(reference_string, kind)`` tuples, deduplicated.
    """
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    for match in _ARXIV_CITE_RE.finditer(text):
        ref = match.group(1)
        if ref not in seen:
            seen.add(ref)
            results.append((ref, "arxiv"))

    for match in _NUMERIC_CITE_RE.finditer(text):
        ref = match.group(1)
        if ref in seen:
            continue
        # Skip if this was already captured as part of an arxiv id
        start = match.start()
        # Check the character before this match to see if it's part of a
        # larger pattern like [2301.12345] where "12345" alone would match
        preceding = text[:start]
        if preceding.endswith("[") is False and _ARXIV_CITE_RE.search(
            text[max(0, start - 12) : match.end() + 1]
        ):
            continue
        seen.add(ref)
        results.append((ref, "numeric"))

    for match in _AUTHOR_YEAR_CITE_RE.finditer(text):
        ref = match.group(1)
        if ref not in seen:
            seen.add(ref)
            results.append((ref, "author_year"))

    return results


def _match_author_year(
    ref: str,
    paper_titles: list[str],
    paper_ids: list[str],
) -> str | None:
    """Try to match an author-year citation to a paper title.

    Extracts the author surname from the reference and checks whether any
    paper title contains it.  Returns the first matching paper_id, or
    ``None`` if no match is found.

    Args:
        ref: Author-year reference string (e.g. ``"Smith et al., 2023"``).
        paper_titles: Ordered list of paper titles.
        paper_ids: Ordered list of paper identifiers (same order).

    Returns:
        Matched paper_id or ``None``.
    """
    # Extract surname (first word before comma or 'et al.')
    surname_match = re.match(r"([A-Za-z'-]+)", ref)
    if not surname_match:
        return None
    surname = surname_match.group(1).lower()

    # Extract year
    year_match = re.search(r"(\d{4})", ref)
    year = year_match.group(1) if year_match else None

    for idx, title in enumerate(paper_titles):
        norm = _normalize_title(title)
        if surname in norm:
            if year is None or year in title:
                return paper_ids[idx]
            # Accept surname match even without year in title
            return paper_ids[idx]

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_fact_score(
    text: str,
    paper_ids: list[str],
    paper_titles: list[str],
) -> FACTScore:
    """Compute FACT citation verification score for a research report.

    Parses ``[n]``, ``[arxiv_id]``, and ``[Author et al., YYYY]`` style
    citations from *text* and matches them against the known paper corpus.

    Args:
        text: Markdown report content.
        paper_ids: Ordered list of paper identifiers (arxiv IDs, DOIs, etc.).
        paper_titles: Ordered list of paper titles (same order as *paper_ids*).

    Returns:
        :class:`FACTScore` with accuracy, coverage, and diagnostic details.

    Raises:
        ValueError: If *paper_ids* and *paper_titles* have different lengths.
    """
    if len(paper_ids) != len(paper_titles):
        raise ValueError(
            f"paper_ids ({len(paper_ids)}) and paper_titles "
            f"({len(paper_titles)}) must have the same length."
        )

    citations = _extract_citations(text)
    logger.debug("Extracted %d unique citations from text", len(citations))

    citation_map: dict[str, str] = {}
    unsupported: list[str] = []
    cited_paper_ids: set[str] = set()
    paper_id_set = set(paper_ids)

    for ref, kind in citations:
        matched_id: str | None = None

        if kind == "numeric":
            idx = int(ref) - 1  # 1-based → 0-based
            if 0 <= idx < len(paper_ids):
                matched_id = paper_ids[idx]
        elif kind == "arxiv":
            if ref in paper_id_set:
                matched_id = ref
            else:
                # Try without version suffix
                bare = re.sub(r"v\d+$", "", ref)
                for pid in paper_ids:
                    if pid == bare or pid.startswith(bare):
                        matched_id = pid
                        break
        elif kind == "author_year":
            matched_id = _match_author_year(ref, paper_titles, paper_ids)

        if matched_id is not None:
            citation_map[ref] = matched_id
            cited_paper_ids.add(matched_id)
        else:
            unsupported.append(ref)

    total = len(citations)
    verified = len(citation_map)
    accuracy = verified / total if total > 0 else 1.0
    corpus_size = len(paper_ids)
    effective_ratio = len(cited_paper_ids) / corpus_size if corpus_size > 0 else 1.0

    uncited = [pid for pid in paper_ids if pid not in cited_paper_ids]

    logger.debug(
        "FACT score: accuracy=%.3f, effective_ratio=%.3f, "
        "verified=%d/%d, uncited=%d",
        accuracy,
        effective_ratio,
        verified,
        total,
        len(uncited),
    )

    return FACTScore(
        citation_accuracy=round(accuracy, 4),
        effective_citation_ratio=round(effective_ratio, 4),
        total_citations=total,
        verified_citations=verified,
        unsupported_citations=unsupported,
        uncited_papers=uncited,
        citation_map=citation_map,
    )
