"""Citation context extraction from converted Markdown papers.

Identifies in-text citations (e.g., [1], [Smith et al., 2024]) and
extracts the surrounding sentences as "citation contexts". These
contexts reveal *how* a paper is being used by the citing paper.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Patterns for in-text citation markers
# Numeric: [1], [1, 2], [1-3], [1,2,3]
_NUMERIC_CITE_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")

# Author-year: (Smith, 2024), (Smith et al., 2024), (Smith & Jones, 2024)
_AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][a-z]+(?:\s+(?:et\s+al\.|&\s+[A-Z][a-z]+))?[,;]\s*\d{4})\)"
)

# Sentence boundary (approximate)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


@dataclass(frozen=True)
class CitationContext:
    """A single citation context extracted from a paper.

    Attributes:
        marker: The citation marker as it appears (e.g., "[1]", "(Smith, 2024)").
        sentence: The sentence containing the citation.
        paragraph: The full paragraph for broader context.
        section: The section heading under which this citation appears.
        position: Character offset of the citation in the document.
    """

    marker: str
    sentence: str
    paragraph: str = ""
    section: str = ""
    position: int = 0


def _split_sentences(text: str) -> list[str]:
    """Split text into approximate sentences."""
    sentences = _SENTENCE_END_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _find_current_section(text: str, position: int) -> str:
    """Find the most recent section heading before position."""
    section_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    current_section = ""
    for match in section_re.finditer(text):
        if match.start() <= position:
            current_section = match.group(2).strip()
        else:
            break
    return current_section


def _is_bibliography_section(section: str) -> bool:
    """Check if a section name is a bibliography/references section."""
    bib_names = {
        "references",
        "bibliography",
        "works cited",
        "literature cited",
        "citations",
    }
    return section.lower().strip() in bib_names


def extract_citation_contexts(
    markdown_text: str,
    context_window: int = 1,
) -> list[CitationContext]:
    """Extract citation contexts from a Markdown document.

    Finds all in-text citations and returns the surrounding sentence(s)
    as context. Skips citations in bibliography/reference sections.

    Args:
        markdown_text: Full Markdown text of the paper.
        context_window: Number of extra sentences before/after the citing
            sentence to include (0 = just the citing sentence).

    Returns:
        List of CitationContext objects.
    """
    if not markdown_text.strip():
        return []

    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", markdown_text)

    contexts: list[CitationContext] = []
    offset = 0

    for paragraph in paragraphs:
        para_text = paragraph.strip()
        if not para_text:
            offset += len(paragraph) + 2
            continue

        section = _find_current_section(markdown_text, offset)

        # Skip bibliography sections
        if _is_bibliography_section(section):
            offset += len(paragraph) + 2
            continue

        # Skip if the paragraph looks like a reference entry
        if re.match(r"^\[\d+\]\s+[A-Z]", para_text):
            offset += len(paragraph) + 2
            continue

        sentences = _split_sentences(para_text)

        for pattern in (_NUMERIC_CITE_RE, _AUTHOR_YEAR_RE):
            for match in pattern.finditer(para_text):
                cite_pos = match.start()
                marker = match.group(0)

                # Find which sentence contains this citation
                citing_sentence = para_text
                char_count = 0
                sentence_idx = 0
                for idx, sent in enumerate(sentences):
                    sent_start = para_text.find(sent, char_count)
                    sent_end = sent_start + len(sent)
                    if sent_start <= cite_pos < sent_end:
                        citing_sentence = sent
                        sentence_idx = idx
                        break
                    char_count = sent_end

                # Build context window
                if context_window > 0 and len(sentences) > 1:
                    start = max(0, sentence_idx - context_window)
                    end = min(len(sentences), sentence_idx + context_window + 1)
                    citing_sentence = " ".join(sentences[start:end])

                ctx = CitationContext(
                    marker=marker,
                    sentence=citing_sentence,
                    paragraph=para_text,
                    section=section,
                    position=offset + cite_pos,
                )
                contexts.append(ctx)

        offset += len(paragraph) + 2

    logger.info("Extracted %d citation contexts", len(contexts))
    return contexts


def group_by_marker(
    contexts: list[CitationContext],
) -> dict[str, list[CitationContext]]:
    """Group citation contexts by their marker.

    Args:
        contexts: List of citation contexts.

    Returns:
        Dict mapping marker string to list of contexts using that marker.
    """
    groups: dict[str, list[CitationContext]] = {}
    for ctx in contexts:
        groups.setdefault(ctx.marker, []).append(ctx)
    return groups


def contexts_to_dicts(
    contexts: list[CitationContext],
) -> list[dict[str, object]]:
    """Serialize citation contexts to a list of dicts for JSON output.

    Args:
        contexts: List of citation contexts.

    Returns:
        List of serializable dicts.
    """
    return [
        {
            "marker": ctx.marker,
            "sentence": ctx.sentence,
            "section": ctx.section,
            "position": ctx.position,
        }
        for ctx in contexts
    ]
