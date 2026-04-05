"""Semantic embedding and scoring using SPECTER2.

Provides embedding computation and cosine-similarity scoring for
candidate papers.  SPECTER2 model loading is lazy and cached.
Falls back gracefully when transformers/torch are not installed.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_model_cache: dict[str, Any] = {}


def _is_specter2_available() -> bool:
    """Check whether the SPECTER2 dependencies are installed."""
    try:
        import adapters  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _load_model(
    model_name: str = "allenai/specter2",
) -> tuple[Any, Any]:
    """Load SPECTER2 model and tokenizer (lazy, cached).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ImportError: If transformers/torch/adapters are not installed.
    """
    cache_key = model_name
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    import adapters  # noqa: F401
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading SPECTER2 model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Load the proximity adapter for similarity tasks
    model.load_adapter(
        "allenai/specter2", source="hf", load_as="specter2_proximity", set_active=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    _model_cache[cache_key] = (model, tokenizer)
    logger.info("SPECTER2 model loaded on %s", device)
    return model, tokenizer


def compute_embeddings(
    texts: list[str],
    model_name: str = "allenai/specter2",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute embeddings for a list of texts using SPECTER2.

    Args:
        texts: Input texts to embed.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.

    Returns:
        numpy array of shape (len(texts), embedding_dim).

    Raises:
        ImportError: If SPECTER2 dependencies are not installed.
    """
    import torch

    model, tokenizer = _load_model(model_name)
    device = next(model.parameters()).device

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and each row of b.

    Args:
        a: Query vector of shape (dim,).
        b: Matrix of shape (n, dim).

    Returns:
        Array of similarities of shape (n,).
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def score_semantic(
    topic: str,
    candidates: list[Any],
    model_name: str = "allenai/specter2",
    batch_size: int = 32,
) -> list[float]:
    """Compute semantic similarity scores between a topic and candidates.

    Args:
        topic: Research topic or query string.
        candidates: List of CandidateRecord objects.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.

    Returns:
        List of similarity scores in [0, 1], one per candidate.
    """
    if not candidates:
        return []

    # Format query in SPECTER2 style
    query_text = f"{topic} [SEP]"
    candidate_texts = [f"{c.title} {c.abstract}" for c in candidates]

    all_texts = [query_text] + candidate_texts
    embeddings = compute_embeddings(
        all_texts, model_name=model_name, batch_size=batch_size
    )

    query_emb = embeddings[0]
    candidate_embs = embeddings[1:]

    raw_scores = _cosine_similarity(query_emb, candidate_embs)

    # Normalize to [0, 1]
    min_s = float(raw_scores.min())
    max_s = float(raw_scores.max())
    if max_s == min_s:
        return [0.5] * len(candidates)

    normalized = (raw_scores - min_s) / (max_s - min_s)
    return [round(float(s), 4) for s in normalized]
