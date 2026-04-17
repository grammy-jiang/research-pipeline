"""Segment-level memory entry management.

Caps episodic and working memory entries at a configurable token limit
(default 450 tokens ≈ ~340 words).  Large entries are split into segments
for better retrieval precision, following the Memory Survey (PVLDB 2026)
and Deep Researcher recommendations.

Each segment carries a ``segment_index`` and ``parent_key`` so the
original entry can be reconstructed when needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 450
_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Estimate token count using the ≈0.75 words-per-token heuristic.

    Conservative: slightly over-counts so segments stay within budget.
    """
    word_count = len(_WORD_RE.findall(text))
    return int(word_count / 0.75) + 1


def segment_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split *text* into segments of at most *max_tokens* estimated tokens.

    Splits on sentence boundaries where possible.  The last
    *overlap_sentences* sentences of each segment are repeated at the
    start of the next to preserve local context.

    Returns a list of segment strings.  If *text* already fits within
    *max_tokens* a single-element list is returned.
    """
    if not text or not text.strip():
        return [text] if text else [""]

    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    segments: list[str] = []
    current_sents: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)

        if current_tokens + sent_tokens > max_tokens and current_sents:
            segments.append(" ".join(current_sents))
            # Carry overlap sentences forward
            overlap = (
                current_sents[-overlap_sentences:] if overlap_sentences > 0 else []
            )
            current_sents = list(overlap)
            current_tokens = sum(estimate_tokens(s) for s in current_sents)

        current_sents.append(sent)
        current_tokens += sent_tokens

    if current_sents:
        segments.append(" ".join(current_sents))

    return segments if segments else [text]


def _split_sentences(text: str) -> list[str]:
    """Rough sentence splitting on period/question/exclamation marks."""
    # Split on sentence-ending punctuation followed by whitespace or end
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class MemorySegment:
    """A bounded segment of a larger memory entry."""

    parent_key: str
    segment_index: int
    total_segments: int
    content: str
    estimated_tokens: int
    metadata: dict[str, str] = field(default_factory=dict)


def segment_entry(
    key: str,
    content: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_sentences: int = 1,
    metadata: dict[str, str] | None = None,
) -> list[MemorySegment]:
    """Segment a memory entry into bounded chunks.

    Parameters
    ----------
    key:
        Identifier for the parent entry.
    content:
        Text content to segment.
    max_tokens:
        Maximum estimated tokens per segment (default 450).
    overlap_sentences:
        Sentences to overlap between segments for continuity.
    metadata:
        Extra metadata to attach to each segment.

    Returns
    -------
    List of :class:`MemorySegment` objects, one per chunk.
    """
    parts = segment_text(
        content, max_tokens=max_tokens, overlap_sentences=overlap_sentences
    )
    total = len(parts)
    extra = metadata or {}

    segments = [
        MemorySegment(
            parent_key=key,
            segment_index=i,
            total_segments=total,
            content=part,
            estimated_tokens=estimate_tokens(part),
            metadata={**extra, "parent_key": key, "segment": f"{i + 1}/{total}"},
        )
        for i, part in enumerate(parts)
    ]

    if total > 1:
        logger.debug(
            "Segmented entry '%s' into %d segments (max_tokens=%d)",
            key,
            total,
            max_tokens,
        )

    return segments
