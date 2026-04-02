"""Topic-aware chunk retrieval using BM25 over chunk index."""

import logging

from rank_bm25 import BM25Okapi

from research_pipeline.models.extraction import ChunkMetadata

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def retrieve_relevant_chunks(
    chunks: list[tuple[ChunkMetadata, str]],
    query_terms: list[str],
    top_k: int = 10,
) -> list[tuple[ChunkMetadata, str, float]]:
    """Retrieve the most relevant chunks for a query using BM25.

    Args:
        chunks: List of (metadata, text) tuples.
        query_terms: Query terms to match against chunks.
        top_k: Maximum chunks to return.

    Returns:
        List of (metadata, text, score) tuples, sorted by relevance.
    """
    if not chunks or not query_terms:
        return []

    texts = [text for _, text in chunks]
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    query_tokenized = _tokenize(" ".join(query_terms))
    scores = bm25.get_scores(query_tokenized)

    scored = [
        (chunks[i][0], chunks[i][1], float(scores[i])) for i in range(len(chunks))
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    result = scored[:top_k]

    logger.info(
        "Retrieved top-%d chunks from %d total (query: %s)",
        len(result),
        len(chunks),
        " ".join(query_terms[:5]),
    )
    return result
