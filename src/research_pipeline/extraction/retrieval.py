"""Topic-aware chunk retrieval using BM25 with optional SPECTER2 hybrid fusion.

When SPECTER2 dependencies are available, combines BM25 keyword matching with
semantic embedding similarity using reciprocal rank fusion (RRF).  Falls back
to BM25-only when transformers/torch are not installed.
"""

import logging

from rank_bm25 import BM25Okapi

from research_pipeline.models.extraction import ChunkMetadata

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _is_embedding_available() -> bool:
    """Check whether SPECTER2 embedding dependencies are installed."""
    try:
        from research_pipeline.screening.embedding import _is_specter2_available

        return _is_specter2_available()
    except ImportError:
        return False


def _bm25_rank(
    chunks: list[tuple[ChunkMetadata, str]],
    query_terms: list[str],
) -> list[tuple[int, float]]:
    """Rank chunks by BM25 score.

    Args:
        chunks: List of (metadata, text) tuples.
        query_terms: Query terms to match.

    Returns:
        List of (index, score) sorted by score descending.
    """
    texts = [text for _, text in chunks]
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    query_tokenized = _tokenize(" ".join(query_terms))
    scores = bm25.get_scores(query_tokenized)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in ranked]


def _embedding_rank(
    chunks: list[tuple[ChunkMetadata, str]],
    query_terms: list[str],
) -> list[tuple[int, float]]:
    """Rank chunks by SPECTER2 embedding similarity.

    Args:
        chunks: List of (metadata, text) tuples.
        query_terms: Query terms for the query embedding.

    Returns:
        List of (index, score) sorted by similarity descending.
    """
    import numpy as np

    from research_pipeline.screening.embedding import (
        _cosine_similarity,
        compute_embeddings,
    )

    query_text = " ".join(query_terms)
    chunk_texts = [text for _, text in chunks]

    all_texts = [query_text] + chunk_texts
    embeddings = compute_embeddings(all_texts, batch_size=32)

    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    raw_scores = _cosine_similarity(query_emb, chunk_embs)

    # Normalize to [0, 1]
    min_s = float(np.min(raw_scores))
    max_s = float(np.max(raw_scores))
    if max_s == min_s:
        normalized = [0.5] * len(chunks)
    else:
        normalized = [(float(s) - min_s) / (max_s - min_s) for s in raw_scores]

    ranked = sorted(enumerate(normalized), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in ranked]


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple rankings using reciprocal rank fusion.

    RRF score = sum(1 / (k + rank_i)) for each ranking.

    Args:
        rankings: List of rankings, each is [(index, score), ...].
        k: RRF constant (default 60, standard value).

    Returns:
        Fused ranking as [(index, rrf_score), ...] sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _original_score) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_relevant_chunks(
    chunks: list[tuple[ChunkMetadata, str]],
    query_terms: list[str],
    top_k: int = 10,
    use_embeddings: bool | None = None,
) -> list[tuple[ChunkMetadata, str, float]]:
    """Retrieve the most relevant chunks for a query.

    When SPECTER2 is available and ``use_embeddings`` is not False,
    combines BM25 with embedding similarity using reciprocal rank fusion.
    Otherwise falls back to BM25-only.

    Args:
        chunks: List of (metadata, text) tuples.
        query_terms: Query terms to match against chunks.
        top_k: Maximum chunks to return.
        use_embeddings: Whether to use SPECTER2 embeddings.
            None = auto-detect (use if available).
            True = force (raises ImportError if unavailable).
            False = disable (BM25 only).

    Returns:
        List of (metadata, text, score) tuples, sorted by relevance.
    """
    if not chunks or not query_terms:
        return []

    # Determine whether to use embeddings
    if use_embeddings is None:
        use_embeddings = _is_embedding_available()
    elif use_embeddings and not _is_embedding_available():
        raise ImportError(
            "SPECTER2 dependencies (transformers, torch, adapters) "
            "are required for embedding retrieval but not installed."
        )

    # BM25 ranking (always computed)
    bm25_ranked = _bm25_rank(chunks, query_terms)

    if use_embeddings:
        try:
            emb_ranked = _embedding_rank(chunks, query_terms)
            fused = _reciprocal_rank_fusion([bm25_ranked, emb_ranked])
            logger.info(
                "Hybrid BM25+embedding retrieval: %d chunks, top-%d",
                len(chunks),
                top_k,
            )
        except Exception as exc:
            logger.warning("Embedding retrieval failed, falling back to BM25: %s", exc)
            fused = [(idx, score) for idx, score in bm25_ranked]
    else:
        fused = [(idx, score) for idx, score in bm25_ranked]
        logger.info(
            "BM25-only retrieval: %d chunks, top-%d",
            len(chunks),
            top_k,
        )

    result = [(chunks[idx][0], chunks[idx][1], score) for idx, score in fused[:top_k]]
    return result
