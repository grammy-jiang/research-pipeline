"""Q2D (Query-to-Document) augmentation for academic search.

Generates hypothetical document snippets that mimic how academic
abstracts are written, converting keyword queries into natural
language pseudo-abstracts.  This bridges the vocabulary gap between
user queries and paper abstracts, boosting recall by 10-25% (Gao et al.,
"Precise Zero-Shot Dense Retrieval without Relevance Labels", 2022).

Also provides domain-aware synonym expansion to cover alternative
terminology used across subfields (e.g., "RAG" → "retrieval augmented
generation", "LLM" → "large language model").
"""

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Domain-aware acronym and synonym mappings.
# Maps common short forms to their expansions so that both appear
# in query variants, covering papers that use either form.
_DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "llm": ["large language model"],
    "llms": ["large language models"],
    "rag": ["retrieval augmented generation"],
    "rlhf": ["reinforcement learning from human feedback"],
    "gpt": ["generative pre-trained transformer"],
    "bert": ["bidirectional encoder representations from transformers"],
    "nlp": ["natural language processing"],
    "cv": ["computer vision"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "cnn": ["convolutional neural network"],
    "rnn": ["recurrent neural network"],
    "lstm": ["long short-term memory"],
    "gan": ["generative adversarial network"],
    "vae": ["variational autoencoder"],
    "moe": ["mixture of experts"],
    "dpo": ["direct preference optimization"],
    "ppo": ["proximal policy optimization"],
    "sft": ["supervised fine-tuning"],
    "cot": ["chain of thought"],
    "tot": ["tree of thought"],
    "icl": ["in-context learning"],
    "peft": ["parameter-efficient fine-tuning"],
    "lora": ["low-rank adaptation"],
    "mcp": ["model context protocol"],
    "kg": ["knowledge graph"],
    "qa": ["question answering"],
    "ner": ["named entity recognition"],
    "pos": ["part of speech"],
    "tf-idf": ["term frequency inverse document frequency"],
    "bm25": ["best matching 25", "okapi bm25"],
    "asr": ["automatic speech recognition"],
    "tts": ["text to speech"],
    "ocr": ["optical character recognition"],
    "gnn": ["graph neural network"],
    "transformer": ["self-attention mechanism"],
    "attention": ["attention mechanism", "self-attention"],
    "fine-tuning": ["fine tuning", "finetuning"],
    "pre-training": ["pre training", "pretraining"],
    "zero-shot": ["zero shot"],
    "few-shot": ["few shot"],
    "multi-modal": ["multimodal"],
    "multi-agent": ["multi agent"],
}

# Q2D templates that mimic different sections of academic papers.
# Each template places the topic in a natural academic context.
_Q2D_TEMPLATES: list[str] = [
    # Abstract opening patterns
    "this paper presents {topic}",
    "we propose a method for {topic}",
    "a survey of {topic}",
    # Methodology patterns
    "we introduce a novel approach to {topic}",
    "our method addresses the challenge of {topic}",
    # Related work patterns
    "recent work on {topic} has shown",
    "existing approaches to {topic} include",
    # Results patterns
    "experiments demonstrate that {topic} achieves",
    "evaluation on {topic} benchmarks shows",
    # Problem statement patterns
    "the problem of {topic} remains challenging",
]


def expand_domain_synonyms(terms: Sequence[str]) -> list[str]:
    """Expand domain-specific acronyms and synonyms in query terms.

    For each term that matches a known acronym or short form, adds the
    expanded form(s) as additional terms.  Original terms are preserved.

    Args:
        terms: Input query terms.

    Returns:
        Expanded list with original terms plus any synonym expansions.
        Duplicates are removed while preserving order.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    for term in terms:
        lower = term.lower().strip()
        if lower and lower not in seen:
            seen.add(lower)
            expanded.append(term)

        # Check for synonym expansion
        synonyms = _DOMAIN_SYNONYMS.get(lower, [])
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower not in seen:
                seen.add(syn_lower)
                expanded.append(syn)

    if len(expanded) > len(terms):
        logger.debug(
            "Domain expansion: %d terms → %d terms",
            len(terms),
            len(expanded),
        )

    return expanded


def generate_q2d_queries(
    must_terms: Sequence[str],
    nice_terms: Sequence[str],
    max_queries: int = 5,
) -> list[str]:
    """Generate Q2D-style pseudo-abstract query variants.

    Creates hypothetical document snippets by inserting the topic
    phrase into academic writing templates.  These pseudo-documents
    better match the vocabulary and structure of real paper abstracts.

    Args:
        must_terms: High-priority query terms.
        nice_terms: Lower-priority query terms.
        max_queries: Maximum number of Q2D queries to generate.

    Returns:
        List of Q2D-augmented query strings.
    """
    if not must_terms and not nice_terms:
        return []

    topic = " ".join(list(must_terms) + list(nice_terms)).strip()
    if not topic:
        return []

    queries: list[str] = []
    seen: set[str] = set()

    for template in _Q2D_TEMPLATES:
        q = template.format(topic=topic)
        if q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= max_queries:
            break

    # Also generate variants with must-terms only for focused matching
    if nice_terms and must_terms:
        must_topic = " ".join(must_terms).strip()
        for template in _Q2D_TEMPLATES[:3]:  # Use first 3 templates
            q = template.format(topic=must_topic)
            if q not in seen:
                seen.add(q)
                queries.append(q)
            if len(queries) >= max_queries:
                break

    return queries[:max_queries]


def augment_query_plan(
    must_terms: list[str],
    nice_terms: list[str],
    existing_variants: list[str],
    max_total_variants: int = 10,
) -> list[str]:
    """Augment an existing query plan with Q2D and synonym expansion.

    Applies domain synonym expansion to terms, then generates Q2D
    queries and merges them with existing variants.  Existing variants
    are preserved (prepended) so the original plan is not lost.

    Args:
        must_terms: High-priority query terms.
        nice_terms: Lower-priority query terms.
        existing_variants: Already-generated query variants.
        max_total_variants: Maximum total variants after augmentation.

    Returns:
        Merged list of query variants (existing + Q2D augmented).
    """
    seen: set[str] = set()
    merged: list[str] = []

    # Keep existing variants first
    for v in existing_variants:
        if v not in seen:
            seen.add(v)
            merged.append(v)

    # Expand domain synonyms
    expanded_must = expand_domain_synonyms(must_terms)
    expanded_nice = expand_domain_synonyms(nice_terms)

    # Add a variant using expanded terms if it's different
    expanded_query = " ".join(expanded_must + expanded_nice).strip()
    if expanded_query and expanded_query not in seen:
        seen.add(expanded_query)
        merged.append(expanded_query)

    # Generate Q2D queries using expanded terms
    remaining = max_total_variants - len(merged)
    if remaining > 0:
        q2d = generate_q2d_queries(expanded_must, expanded_nice, max_queries=remaining)
        for q in q2d:
            if q not in seen:
                seen.add(q)
                merged.append(q)
            if len(merged) >= max_total_variants:
                break

    return merged[:max_total_variants]
