"""Paper similarity clustering using TF-IDF and agglomerative grouping.

Clusters screened candidates by title+abstract similarity so that
downstream synthesis can process topically coherent groups.

Deep Research Report §B5: Paper similarity clustering.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from research_pipeline.models.candidate import CandidateRecord

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z]{2,}")
_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "been",
        "our",
        "their",
        "which",
        "can",
        "not",
        "but",
        "also",
        "more",
        "than",
        "its",
        "each",
        "such",
        "into",
        "over",
        "both",
        "these",
        "those",
        "using",
        "based",
        "show",
        "results",
        "paper",
        "propose",
        "method",
        "approach",
        "model",
        "use",
        "used",
        "however",
    }
)


@dataclass
class PaperCluster:
    """A cluster of topically related papers."""

    cluster_id: int
    label: str
    paper_ids: list[str] = field(default_factory=list)
    top_terms: list[str] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization with stop-word removal."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP_WORDS]


def _build_tfidf(
    docs: list[list[str]],
) -> tuple[list[str], list[dict[str, float]]]:
    """Build TF-IDF vectors for a list of tokenized documents.

    Returns:
        Tuple of (vocabulary list, list of {term: tfidf_weight} dicts).
    """
    n_docs = len(docs)
    if n_docs == 0:
        return [], []

    # Document frequency
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    vocab = sorted(df.keys())
    idf = {term: math.log(1 + n_docs / (1 + count)) for term, count in df.items()}

    vectors: list[dict[str, float]] = []
    for doc in docs:
        tf = Counter(doc)
        total = len(doc) if doc else 1
        vec = {term: (count / total) * idf[term] for term, count in tf.items()}
        vectors.append(vec)
    return vocab, vectors


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _agglomerative_cluster(
    vectors: list[dict[str, float]],
    threshold: float,
) -> list[int]:
    """Simple single-linkage agglomerative clustering.

    Args:
        vectors: TF-IDF sparse vectors per document.
        threshold: Minimum cosine similarity to merge clusters.

    Returns:
        List of cluster IDs (one per document).
    """
    n = len(vectors)
    if n == 0:
        return []

    # Each document starts in its own cluster
    labels = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            if sim >= threshold:
                # Merge: relabel all j-cluster docs to i-cluster
                old_label = labels[j]
                new_label = labels[i]
                if old_label != new_label:
                    for k in range(n):
                        if labels[k] == old_label:
                            labels[k] = new_label

    # Compact labels to 0..K-1
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[lbl] for lbl in labels]


def _cluster_top_terms(
    docs: list[list[str]],
    labels: list[int],
    cluster_id: int,
    top_n: int = 5,
) -> list[str]:
    """Get top TF terms for a cluster."""
    combined: Counter[str] = Counter()
    for i, lbl in enumerate(labels):
        if lbl == cluster_id:
            combined.update(docs[i])
    return [term for term, _ in combined.most_common(top_n)]


def _cluster_label(top_terms: list[str]) -> str:
    """Generate a human-readable cluster label from top terms."""
    return ", ".join(top_terms[:3]) if top_terms else "misc"


def cluster_candidates(
    candidates: list[CandidateRecord],
    threshold: float = 0.15,
) -> list[PaperCluster]:
    """Cluster candidates by title+abstract TF-IDF similarity.

    Args:
        candidates: List of candidate records to cluster.
        threshold: Cosine similarity threshold for merging (0-1).
            Lower values produce fewer, larger clusters.

    Returns:
        List of PaperCluster objects, sorted by cluster size descending.
    """
    if not candidates:
        return []

    # Build document representations
    docs = [_tokenize(f"{c.title} {c.abstract}") for c in candidates]
    _vocab, vectors = _build_tfidf(docs)
    labels = _agglomerative_cluster(vectors, threshold)

    n_clusters = max(labels) + 1 if labels else 0
    clusters: list[PaperCluster] = []

    for cid in range(n_clusters):
        paper_ids = [
            candidates[i].arxiv_id for i, lbl in enumerate(labels) if lbl == cid
        ]
        top_terms = _cluster_top_terms(docs, labels, cid)
        label = _cluster_label(top_terms)
        clusters.append(
            PaperCluster(
                cluster_id=cid,
                label=label,
                paper_ids=paper_ids,
                top_terms=top_terms,
            )
        )

    # Sort by size descending
    clusters.sort(key=lambda c: len(c.paper_ids), reverse=True)
    logger.info(
        "Clustered %d papers into %d groups (threshold=%.2f)",
        len(candidates),
        len(clusters),
        threshold,
    )
    return clusters
