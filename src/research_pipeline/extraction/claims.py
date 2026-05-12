"""Heuristic claim extraction from Markdown paper chunks.

Extracts candidate claims from key sections (Abstract, Conclusion, Results,
Discussion) without requiring an LLM. Each claim is a declarative sentence
from a high-signal section, with supporting chunk IDs retrieved via BM25.

LLM-based extraction (higher precision, v2+) will replace or extend this
module when an LLM provider is configured.
"""

from __future__ import annotations

import re

from research_pipeline.extraction.retrieval import retrieve_relevant_chunks
from research_pipeline.models.extraction import ChunkMetadata, ExtractedClaim

# Sections with high claim density
_HIGH_SIGNAL_SECTIONS = re.compile(
    r"\b(abstract|conclusion|summary|result|finding|contribution|discussion"
    r"|implication|recommendation|outcome|evaluation|experiment)\b",
    re.IGNORECASE,
)

# Patterns that indicate a declarative assertion (not questions, not "we present")
_ASSERTION_PATTERN = re.compile(
    r"\b(shows?|demonstrates?|proves?|establishes?|confirms?|achieves?|improves?"
    r"|outperforms?|reduces?|increases?|enables?|proposes?|introduces?|presents?"
    r"|develops?|designs?|is|are|was|were|has|have|can|will|provides?|offers?"
    r"|results? in)\b",
    re.IGNORECASE,
)

# Patterns to skip (definitions, questions, passive setup phrases)
_SKIP_PATTERN = re.compile(
    r"^(we (describe|present|propose|introduce|outline|discuss|review"
    r"|compare|evaluate|consider)|in (this|the) (paper|section|work|study)"
    r"|figure \d|table \d|appendix|for (example|instance)|note that"
    r"|see (also|section)|\[\d+\])",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering noise."""
    # Remove markdown artifacts (citations, URLs, footnotes)
    text = re.sub(r"\[[\d,\s–-]+\]", "", text)  # citation brackets
    text = re.sub(r"https?://\S+", "", text)  # URLs
    text = re.sub(r"\s+", " ", text).strip()

    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]


def _is_claim_sentence(sentence: str) -> bool:
    """Return True if the sentence looks like an assertable claim."""
    if len(sentence) < 30 or len(sentence) > 400:
        return False
    if sentence.endswith("?"):
        return False
    if _SKIP_PATTERN.match(sentence):
        return False
    return bool(_ASSERTION_PATTERN.search(sentence))


def _section_confidence(section_path: str) -> float:
    """Base confidence derived from section type."""
    section_lower = section_path.lower()
    if re.search(r"\babstract\b", section_lower):
        return 0.85
    if re.search(r"\b(conclusion|summary|implication)s?\b", section_lower):
        return 0.80
    if re.search(r"\b(result|finding|contribution)s?\b", section_lower):
        return 0.75
    if re.search(r"\b(discussion|evaluation|experiment)s?\b", section_lower):
        return 0.65
    return 0.50


def extract_claims_heuristic(
    chunks: list[tuple[ChunkMetadata, str]],
    max_claims: int = 20,
    max_claims_per_section: int = 3,
    top_k_support: int = 3,
) -> list[ExtractedClaim]:
    """Extract candidate claims heuristically from paper chunks.

    Scans chunks from high-signal sections (Abstract, Conclusion, Results),
    extracts declarative sentences, and finds supporting chunk IDs via BM25.

    Args:
        chunks: List of (ChunkMetadata, text) pairs from the paper.
        max_claims: Maximum number of claims to return.
        max_claims_per_section: Maximum claims extracted per section.
        top_k_support: Number of supporting chunks to retrieve per claim.

    Returns:
        List of ExtractedClaim objects, ordered by confidence descending.
    """
    if not chunks:
        return []

    candidates: list[tuple[str, float, str]] = []  # (text, base_conf, chunk_id)

    section_claim_counts: dict[str, int] = {}

    for meta, text in chunks:
        section = meta.section_path
        if not _HIGH_SIGNAL_SECTIONS.search(section):
            continue
        if section_claim_counts.get(section, 0) >= max_claims_per_section:
            continue

        base_conf = _section_confidence(section)
        sentences = _split_sentences(text)
        for sentence in sentences:
            if _is_claim_sentence(sentence):
                candidates.append((sentence, base_conf, meta.chunk_id))
                section_claim_counts[section] = section_claim_counts.get(section, 0) + 1
                if section_claim_counts[section] >= max_claims_per_section:
                    break

    if not candidates:
        return []

    # Sort by base confidence and deduplicate similar claims
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = _deduplicate(candidates)
    candidates = candidates[:max_claims]

    # Build ExtractedClaim objects with supporting chunk IDs via BM25
    claims: list[ExtractedClaim] = []
    for claim_text, base_conf, primary_chunk_id in candidates:
        query_terms = claim_text.lower().split()
        relevant = retrieve_relevant_chunks(
            chunks,
            query_terms,
            top_k=top_k_support,
            use_embeddings=False,
            use_cross_encoder=False,
        )

        # Collect support chunk IDs, boosting confidence if primary chunk is retrieved
        chunk_ids: list[str] = []
        max_retrieval_score = 0.0
        has_primary = False

        for r_meta, _r_text, r_score in relevant:
            chunk_ids.append(r_meta.chunk_id)
            if r_score > max_retrieval_score:
                max_retrieval_score = r_score
            if r_meta.chunk_id == primary_chunk_id:
                has_primary = True

        # Include primary chunk if not already retrieved
        if primary_chunk_id not in chunk_ids:
            chunk_ids.insert(0, primary_chunk_id)

        # Adjust confidence by retrieval strength
        retrieval_boost = min(max_retrieval_score * 0.15, 0.10)
        primary_boost = 0.05 if has_primary else 0.0
        confidence = round(min(base_conf + retrieval_boost + primary_boost, 0.95), 4)

        claims.append(
            ExtractedClaim(
                claim=claim_text,
                chunk_ids=chunk_ids,
                confidence=confidence,
            )
        )

    return sorted(claims, key=lambda c: c.confidence, reverse=True)


def _deduplicate(
    candidates: list[tuple[str, float, str]],
    similarity_threshold: float = 0.6,
) -> list[tuple[str, float, str]]:
    """Remove near-duplicate claims using Jaccard word overlap."""
    result: list[tuple[str, float, str]] = []
    seen_tokens: list[frozenset[str]] = []

    for text, conf, chunk_id in candidates:
        tokens = frozenset(text.lower().split())
        duplicate = False
        for seen in seen_tokens:
            if not seen or not tokens:
                continue
            intersection = len(tokens & seen)
            union = len(tokens | seen)
            if union > 0 and intersection / union >= similarity_threshold:
                duplicate = True
                break
        if not duplicate:
            result.append((text, conf, chunk_id))
            seen_tokens.append(tokens)

    return result
