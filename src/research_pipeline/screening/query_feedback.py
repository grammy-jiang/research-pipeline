"""Query refinement feedback based on screening results.

Analyzes term overlap between the original query and top-K screened
papers to suggest refined queries for iterative search improvement.
Based on SiRe's principle: remove noise before adding signal.
"""

import re
from collections import Counter

from pydantic import BaseModel, Field

from research_pipeline.models.candidate import CandidateRecord
from research_pipeline.models.query_plan import QueryPlan

# Common English academic stopwords to exclude from term analysis.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "approach",
        "are",
        "as",
        "at",
        "based",
        "be",
        "been",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "data",
        "do",
        "each",
        "even",
        "first",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "however",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "method",
        "model",
        "more",
        "most",
        "new",
        "no",
        "not",
        "of",
        "on",
        "one",
        "only",
        "or",
        "other",
        "our",
        "over",
        "paper",
        "proposed",
        "propose",
        "result",
        "results",
        "several",
        "show",
        "shown",
        "some",
        "study",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "two",
        "under",
        "up",
        "upon",
        "us",
        "use",
        "used",
        "using",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, keeping only alphabetic tokens of length > 1.

    Args:
        text: Raw text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def _tokenize_paper(paper: CandidateRecord) -> set[str]:
    """Extract unique non-stopword tokens from a paper's title and abstract.

    Args:
        paper: A candidate paper record.

    Returns:
        Set of unique lowercase tokens (stopwords removed).
    """
    tokens = _tokenize(paper.title) + _tokenize(paper.abstract)
    return {t for t in tokens if t not in STOPWORDS}


class QueryRefinement(BaseModel):
    """Refinement suggestions produced by analyzing query-result overlap."""

    original_query: str = Field(description="The original raw query topic.")
    suggested_additions: list[str] = Field(
        default_factory=list,
        description=(
            "High-signal terms from top-K papers not present in the original query."
        ),
    )
    suggested_removals: list[str] = Field(
        default_factory=list,
        description=("Query terms that appear in very few top-K papers."),
    )
    term_coverage: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "For each query term, the fraction of top-K papers that contain it."
        ),
    )
    emergent_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Frequent terms in top-K papers absent from the query "
            "(potential new topics)."
        ),
    )
    refined_must_terms: list[str] = Field(
        default_factory=list,
        description="Suggested improved must_terms list.",
    )
    refined_nice_terms: list[str] = Field(
        default_factory=list,
        description="Suggested improved nice_terms list.",
    )


def compute_query_refinement(
    query_plan: QueryPlan,
    top_k_papers: list[CandidateRecord],
    k_threshold: float = 0.3,
) -> QueryRefinement:
    """Analyze term overlap and suggest query refinements.

    Args:
        query_plan: The structured query plan from the plan stage.
        top_k_papers: Top-K papers from screening (shortlisted).
        k_threshold: Minimum fraction of top-K papers a query term must
            appear in to be retained. Terms below this are suggested for
            removal.

    Returns:
        A ``QueryRefinement`` with coverage stats and suggestions.
    """
    if not top_k_papers:
        return QueryRefinement(original_query=query_plan.topic_raw)

    # 1. Tokenize all papers.
    paper_token_sets: list[set[str]] = [_tokenize_paper(p) for p in top_k_papers]
    n_papers = len(top_k_papers)

    # 2. Document frequency for every term across top-K.
    doc_freq: Counter[str] = Counter()
    for token_set in paper_token_sets:
        for token in token_set:
            doc_freq[token] += 1

    # Collect all query terms (must + nice), lowercased.
    all_query_terms: list[str] = [
        t.lower() for t in query_plan.must_terms + query_plan.nice_terms
    ]

    # 3. Term coverage: fraction of papers containing each query term.
    term_coverage: dict[str, float] = {}
    for term in all_query_terms:
        term_lower = term.lower()
        term_tokens = _tokenize(term_lower)
        if not term_tokens:
            term_coverage[term] = 0.0
            continue
        # A paper "contains" the term if *all* tokens of the term appear.
        count = sum(
            1 for ts in paper_token_sets if all(tok in ts for tok in term_tokens)
        )
        term_coverage[term] = count / n_papers

    # 4. Suggested removals: query terms below k_threshold.
    suggested_removals: list[str] = [
        term for term in all_query_terms if term_coverage.get(term, 0.0) < k_threshold
    ]

    # 5. Emergent terms: high-DF terms not in the query (top 10 by DF).
    query_token_set: set[str] = set()
    for term in all_query_terms:
        query_token_set.update(_tokenize(term.lower()))
    # Also include tokens from the raw topic.
    query_token_set.update(
        t for t in _tokenize(query_plan.topic_raw) if t not in STOPWORDS
    )

    emergent_candidates = [
        (tok, freq)
        for tok, freq in doc_freq.most_common()
        if tok not in query_token_set and tok not in STOPWORDS and len(tok) > 3
    ]
    emergent_terms = [tok for tok, _ in emergent_candidates[:10]]

    # 6. Suggested additions: top 5 emergent terms.
    suggested_additions = emergent_terms[:5]

    # 7. Build refined must_terms and nice_terms.
    removal_set = {r.lower() for r in suggested_removals}

    refined_must_terms = [
        t for t in query_plan.must_terms if t.lower() not in removal_set
    ]
    refined_nice_terms = [
        t for t in query_plan.nice_terms if t.lower() not in removal_set
    ]

    # Promote top emergent terms: first 2 to must, next 3 to nice.
    additions_for_must = [
        t for t in suggested_additions[:2] if t not in refined_must_terms
    ]
    additions_for_nice = [
        t
        for t in suggested_additions[2:5]
        if t not in refined_nice_terms and t not in refined_must_terms
    ]
    refined_must_terms.extend(additions_for_must)
    refined_nice_terms.extend(additions_for_nice)

    return QueryRefinement(
        original_query=query_plan.topic_raw,
        suggested_additions=suggested_additions,
        suggested_removals=suggested_removals,
        term_coverage=term_coverage,
        emergent_terms=emergent_terms,
        refined_must_terms=refined_must_terms,
        refined_nice_terms=refined_nice_terms,
    )
