"""RACE report quality scoring.

Evaluates research reports across four dimensions:
- **R**eadability: sentence length, paragraph structure, heading density, lists
- **A**ctionability: recommendation keywords, specific findings, practical sections
- **C**omprehensiveness: section coverage, word count, citation diversity
- **E**vidence: citation density, blockquotes, confidence annotations, evidence map
"""

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class RACEScore(BaseModel):
    """Composite RACE score for a research report.

    Each dimension is in [0, 1].  ``overall`` is the equal-weight average.
    ``details`` preserves raw metrics for transparency.
    """

    readability: float = Field(ge=0.0, le=1.0)
    actionability: float = Field(ge=0.0, le=1.0)
    comprehensiveness: float = Field(ge=0.0, le=1.0)
    evidence: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    details: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_EVIDENCE_CITATION_RE = re.compile(r"\[[\w.\-]+\]")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s", re.MULTILINE)
_CONFIDENCE_RE = re.compile(
    r"(🟢|🟡|🔴|high confidence|medium confidence|low confidence"
    r"|confidence.*?:.*?(high|medium|low))",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+\.?\d*\s*%|\d+\.\d+|\b\d{2,}\b")

ACTIONABLE_KEYWORDS = [
    "should",
    "recommend",
    "consider",
    "suggests",
    "propose",
    "implement",
    "adopt",
    "use",
    "apply",
]

REQUIRED_SECTIONS = [
    "executive summary",
    "research question",
    "methodology",
    "papers reviewed",
    "research landscape",
    "research gaps",
    "practical recommendations",
    "references",
    "appendix",
]


def _word_count(text: str) -> int:
    """Return word count of *text*."""
    return len(text.split())


def _paragraphs(text: str) -> list[str]:
    """Split *text* into non-empty paragraphs (blank-line delimited)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sentences(text: str) -> list[str]:
    """Split *text* into sentences (rough heuristic)."""
    # Collapse whitespace and split on sentence-ending punctuation.
    cleaned = re.sub(r"\s+", " ", text)
    parts = _SENTENCE_SPLIT.split(cleaned)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------


def score_readability(text: str) -> tuple[float, dict[str, object]]:
    """Score readability of *text*.

    Heuristics:
    - Average sentence length (target 15-25 words)
    - Paragraph count and average paragraph length
    - Heading density (headings per 1000 words — target 3-8)
    - Use of bullet / numbered lists

    Args:
        text: Markdown report content.

    Returns:
        ``(score, details)`` where *score* is in [0, 1].
    """
    words = _word_count(text)
    sentences = _sentences(text)
    paragraphs = _paragraphs(text)
    heading_count = len(_HEADING_RE.findall(text))
    bullet_count = len(_BULLET_RE.findall(text))
    numbered_count = len(_NUMBERED_RE.findall(text))

    # --- sentence length sub-score ---
    avg_sentence_len = words / len(sentences) if sentences else 0.0

    if 15 <= avg_sentence_len <= 25:
        sentence_score = 1.0
    elif avg_sentence_len == 0:
        sentence_score = 0.0
    else:
        # Linear decay outside target window
        if avg_sentence_len < 15:
            sentence_score = max(0.0, avg_sentence_len / 15)
        else:
            sentence_score = max(0.0, 1.0 - (avg_sentence_len - 25) / 25)

    # --- paragraph sub-score ---
    para_count = len(paragraphs)
    avg_para_len = words / para_count if para_count else 0.0
    para_score = min(1.0, para_count / 10)  # at least 10 paragraphs = full score

    # --- heading density sub-score ---
    heading_density = (heading_count / words * 1000) if words else 0.0
    if 3 <= heading_density <= 8:
        heading_score = 1.0
    elif heading_density == 0:
        heading_score = 0.0
    else:
        if heading_density < 3:
            heading_score = max(0.0, heading_density / 3)
        else:
            heading_score = max(0.0, 1.0 - (heading_density - 8) / 8)

    # --- list usage sub-score ---
    list_items = bullet_count + numbered_count
    list_score = min(1.0, list_items / 5)  # ≥5 list items = full score

    score = (
        0.30 * sentence_score
        + 0.25 * para_score
        + 0.25 * heading_score
        + 0.20 * list_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    details: dict[str, object] = {
        "word_count": words,
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_len, 2),
        "paragraph_count": para_count,
        "avg_paragraph_length": round(avg_para_len, 2),
        "heading_count": heading_count,
        "heading_density_per_1000": round(heading_density, 2),
        "bullet_items": bullet_count,
        "numbered_items": numbered_count,
        "sentence_score": round(sentence_score, 4),
        "paragraph_score": round(para_score, 4),
        "heading_score": round(heading_score, 4),
        "list_score": round(list_score, 4),
    }
    return score, details


def score_actionability(text: str) -> tuple[float, dict[str, object]]:
    """Score actionability of *text*.

    Heuristics:
    - Density of actionable recommendation keywords
    - Lines with numbers / percentages / measurements
    - Presence of "practical recommendations" section

    Args:
        text: Markdown report content.

    Returns:
        ``(score, details)`` where *score* is in [0, 1].
    """
    words = _word_count(text)
    text_lower = text.lower()

    # Actionable keywords
    keyword_counts: dict[str, int] = {}
    total_keywords = 0
    for kw in ACTIONABLE_KEYWORDS:
        count = len(re.findall(rf"\b{kw}\b", text_lower))
        keyword_counts[kw] = count
        total_keywords += count

    keyword_density = (total_keywords / words * 100) if words else 0.0
    # Target: ~1-3% keyword density → full score
    keyword_score = min(1.0, keyword_density / 1.5)

    # Specific findings (lines with numbers/percentages)
    findings_count = 0
    for line in text.splitlines():
        if _NUMBER_RE.search(line):
            findings_count += 1
    findings_density = (findings_count / max(words, 1)) * 100
    findings_score = min(1.0, findings_density / 2.0)

    # Practical recommendations section
    headings = [
        line.strip().lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ]
    has_practical = any("practical recommendations" in h for h in headings)
    practical_score = 1.0 if has_practical else 0.0

    score = 0.40 * keyword_score + 0.30 * findings_score + 0.30 * practical_score
    score = round(max(0.0, min(1.0, score)), 4)

    details: dict[str, object] = {
        "actionable_keyword_total": total_keywords,
        "actionable_keyword_density_pct": round(keyword_density, 2),
        "keyword_counts": keyword_counts,
        "findings_line_count": findings_count,
        "has_practical_recommendations_section": has_practical,
        "keyword_score": round(keyword_score, 4),
        "findings_score": round(findings_score, 4),
        "practical_score": round(practical_score, 4),
    }
    return score, details


def score_comprehensiveness(text: str) -> tuple[float, dict[str, object]]:
    """Score comprehensiveness of *text*.

    Heuristics:
    - Section count vs expected sections
    - Word count (target 3000-15000)
    - Evidence citations per section
    - Diversity of cited papers

    Args:
        text: Markdown report content.

    Returns:
        ``(score, details)`` where *score* is in [0, 1].
    """
    words = _word_count(text)
    headings = [
        line.strip().lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ]
    section_count = len(headings)

    # Section coverage sub-score
    present = sum(1 for req in REQUIRED_SECTIONS if any(req in h for h in headings))
    section_score = present / max(len(REQUIRED_SECTIONS), 1)

    # Word count sub-score (target 3000-15000)
    if 3000 <= words <= 15000:
        word_score = 1.0
    elif words < 3000:
        word_score = max(0.0, words / 3000)
    else:
        # Gentle penalty above 15000 — still decent up to ~30000
        word_score = max(0.0, 1.0 - (words - 15000) / 30000)

    # Evidence citations
    citations = _EVIDENCE_CITATION_RE.findall(text)
    citation_count = len(citations)
    citations_per_section = citation_count / max(section_count, 1)
    citation_score = min(1.0, citations_per_section / 3)  # target ≥3 per section

    # Citation diversity (unique cited papers)
    unique_citations = len(set(citations))
    diversity_score = min(1.0, unique_citations / 5)  # target ≥5 unique refs

    score = (
        0.30 * section_score
        + 0.25 * word_score
        + 0.25 * citation_score
        + 0.20 * diversity_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    details: dict[str, object] = {
        "section_count": section_count,
        "required_sections_present": present,
        "required_sections_total": len(REQUIRED_SECTIONS),
        "word_count": words,
        "citation_count": citation_count,
        "citations_per_section": round(citations_per_section, 2),
        "unique_citations": unique_citations,
        "section_score": round(section_score, 4),
        "word_score": round(word_score, 4),
        "citation_score": round(citation_score, 4),
        "diversity_score": round(diversity_score, 4),
    }
    return score, details


def score_evidence(text: str) -> tuple[float, dict[str, object]]:
    """Score evidence quality of *text*.

    Heuristics:
    - Citation density (citations per 500 words)
    - Blockquote / evidence quote patterns
    - Confidence annotations per claim
    - Evidence map section present

    Args:
        text: Markdown report content.

    Returns:
        ``(score, details)`` where *score* is in [0, 1].
    """
    words = _word_count(text)
    citations = _EVIDENCE_CITATION_RE.findall(text)
    citation_count = len(citations)

    # Citation density (per 500 words; target ≥3)
    citation_density = (citation_count / max(words, 1)) * 500
    density_score = min(1.0, citation_density / 3)

    # Blockquotes
    blockquote_count = len(_BLOCKQUOTE_RE.findall(text))
    blockquote_score = min(1.0, blockquote_count / 3)  # target ≥3

    # Confidence annotations
    confidence_matches = _CONFIDENCE_RE.findall(text)
    confidence_count = len(confidence_matches)
    confidence_score = min(1.0, confidence_count / 5)  # target ≥5

    # Evidence map section
    headings = [
        line.strip().lstrip("#").strip().lower()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ]
    has_evidence_map = any("evidence map" in h for h in headings)
    evidence_map_score = 1.0 if has_evidence_map else 0.0

    score = (
        0.35 * density_score
        + 0.20 * blockquote_score
        + 0.25 * confidence_score
        + 0.20 * evidence_map_score
    )
    score = round(max(0.0, min(1.0, score)), 4)

    details: dict[str, object] = {
        "citation_count": citation_count,
        "citation_density_per_500w": round(citation_density, 2),
        "blockquote_count": blockquote_count,
        "confidence_annotation_count": confidence_count,
        "has_evidence_map_section": has_evidence_map,
        "density_score": round(density_score, 4),
        "blockquote_score": round(blockquote_score, 4),
        "confidence_score": round(confidence_score, 4),
        "evidence_map_score": round(evidence_map_score, 4),
    }
    return score, details


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def compute_race_score(text: str) -> RACEScore:
    """Compute the full RACE score for a research report.

    Equal-weight average of the four dimensions.

    Args:
        text: Markdown report content.

    Returns:
        :class:`RACEScore` with per-dimension scores and details.
    """
    r_score, r_details = score_readability(text)
    a_score, a_details = score_actionability(text)
    c_score, c_details = score_comprehensiveness(text)
    e_score, e_details = score_evidence(text)

    overall = round((r_score + a_score + c_score + e_score) / 4, 4)

    return RACEScore(
        readability=r_score,
        actionability=a_score,
        comprehensiveness=c_score,
        evidence=e_score,
        overall=overall,
        details={
            "readability": r_details,
            "actionability": a_details,
            "comprehensiveness": c_details,
            "evidence": e_details,
        },
    )
