"""Cross-encoder passage reranking for chunk retrieval.

Uses a cross-encoder model (e.g. ms-marco-MiniLM-L-6-v2) to rerank
query-passage pairs for higher precision than bi-encoder or BM25 approaches.
Cross-encoders score each (query, passage) pair directly, producing more
accurate relevance estimates at the cost of higher latency.

Requires ``sentence-transformers`` as an optional dependency.
"""

import logging

from research_pipeline.models.extraction import ChunkMetadata

logger = logging.getLogger(__name__)


def _is_cross_encoder_available() -> bool:
    """Check whether ``sentence-transformers`` is installed.

    Returns:
        True if the CrossEncoder class can be imported.
    """
    try:
        from sentence_transformers import (  # type: ignore[import-not-found]
            CrossEncoder,  # noqa: F401
        )

        return True
    except ImportError:
        return False


def cross_encoder_rerank(
    query: str,
    chunks: list[tuple[ChunkMetadata, str]],
    top_k: int = 10,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[tuple[int, float]]:
    """Rerank chunks using a cross-encoder model.

    Scores each (query, chunk_text) pair directly and returns the top-k
    results sorted by descending score.

    Args:
        query: The search query string.
        chunks: List of (metadata, text) tuples to rerank.
        top_k: Maximum number of results to return.
        model_name: HuggingFace cross-encoder model identifier.

    Returns:
        List of (original_index, score) tuples sorted by score descending,
        limited to top_k entries.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    if not chunks or not query:
        return []

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for cross-encoder reranking "
            "but not installed. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model = CrossEncoder(model_name)

    pairs = [(query, text) for _meta, text in chunks]
    scores = model.predict(pairs)

    indexed_scores = list(enumerate(float(s) for s in scores))
    ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

    logger.info(
        "Cross-encoder reranked %d chunks with model %s, returning top-%d",
        len(chunks),
        model_name,
        min(top_k, len(ranked)),
    )

    return ranked[:top_k]
