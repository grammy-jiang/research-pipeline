"""Topic-aware chunk retrieval with optional SPECTER2 fusion and cross-encoder.

When SPECTER2 dependencies are available, combines BM25 keyword matching with
semantic embedding similarity using reciprocal rank fusion (RRF).  Falls back
to BM25-only when transformers/torch are not installed.

When ``sentence-transformers`` is available, an optional cross-encoder reranking
step rescores the top candidates for higher precision.
"""

import logging

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

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


def _is_cross_encoder_available() -> bool:
    """Check whether cross-encoder reranking dependencies are installed."""
    try:
        from research_pipeline.extraction.cross_encoder import (
            _is_cross_encoder_available as is_available,
        )

        return is_available()
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

    all_texts = [query_text, *chunk_texts]
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
    use_cross_encoder: bool | None = None,
) -> list[tuple[ChunkMetadata, str, float]]:
    """Retrieve the most relevant chunks for a query.

    When SPECTER2 is available and ``use_embeddings`` is not False,
    combines BM25 with embedding similarity using reciprocal rank fusion.
    Otherwise falls back to BM25-only.

    When ``sentence-transformers`` is available and ``use_cross_encoder``
    is not False, the top candidates from BM25 (or hybrid) retrieval are
    reranked using a cross-encoder model for higher precision.

    Args:
        chunks: List of (metadata, text) tuples.
        query_terms: Query terms to match against chunks.
        top_k: Maximum chunks to return.
        use_embeddings: Whether to use SPECTER2 embeddings.
            None = auto-detect (use if available).
            True = force (raises ImportError if unavailable).
            False = disable (BM25 only).
        use_cross_encoder: Whether to use cross-encoder reranking.
            None = auto-detect (use if available and ≤100 chunks).
            True = force (raises ImportError if unavailable).
            False = disable.

    Returns:
        List of (metadata, text, score) tuples, sorted by relevance.
    """
    if not chunks or not query_terms:
        return []

    from research_pipeline.extraction.cross_encoder import cross_encoder_rerank

    # Determine whether to use embeddings
    if use_embeddings is None:
        use_embeddings = _is_embedding_available()
    elif use_embeddings and not _is_embedding_available():
        raise ImportError(
            "SPECTER2 dependencies (transformers, torch, adapters) "
            "are required for embedding retrieval but not installed."
        )

    # Determine whether to use cross-encoder reranking
    _cross_encoder_enabled: bool
    if use_cross_encoder is None:
        _cross_encoder_enabled = _is_cross_encoder_available() and len(chunks) <= 100
    elif use_cross_encoder:
        if not _is_cross_encoder_available():
            raise ImportError(
                "sentence-transformers is required for cross-encoder "
                "reranking but not installed. Install it with: "
                "pip install sentence-transformers"
            )
        _cross_encoder_enabled = True
    else:
        _cross_encoder_enabled = False

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

    # Cross-encoder reranking on top candidates
    if _cross_encoder_enabled:
        # Take a wider candidate pool for reranking
        candidate_limit = max(top_k * 3, 30)
        candidate_indices = [idx for idx, _score in fused[:candidate_limit]]
        candidate_chunks = [chunks[idx] for idx in candidate_indices]
        query_str = " ".join(query_terms)

        try:
            reranked = cross_encoder_rerank(
                query=query_str,
                chunks=candidate_chunks,
                top_k=top_k,
            )
            # Map back to original indices
            final_ranked = [
                (candidate_indices[local_idx], score) for local_idx, score in reranked
            ]
            logger.info(
                "Cross-encoder reranked %d candidates → top-%d",
                len(candidate_chunks),
                top_k,
            )
        except Exception as exc:
            logger.warning(
                "Cross-encoder reranking failed, using pre-rerank ordering: %s",
                exc,
            )
            final_ranked = fused[:top_k]
    else:
        final_ranked = fused[:top_k]

    result = [(chunks[idx][0], chunks[idx][1], score) for idx, score in final_ranked]
    return result
