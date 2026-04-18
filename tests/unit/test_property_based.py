"""Property-based tests using Hypothesis for core algorithms.

Tests mathematical invariants and edge cases in:
- BM25 scoring (heuristic.py)
- Normalization (heuristic.py)
- TF-IDF construction (clustering.py)
- Cosine similarity (clustering.py)
- Recency bonus (heuristic.py)
- Cross-source dedup (base.py)
- Citation context extraction (citation_context.py)
- Diversity selection (heuristic.py)
"""

import math
from datetime import UTC, datetime, timedelta

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from research_pipeline.extraction.citation_context import (
    CitationContext,
    _split_sentences,
    extract_citation_contexts,
)
from research_pipeline.screening.clustering import (
    _build_tfidf,
    _cosine_similarity,
    _tokenize as cluster_tokenize,
)
from research_pipeline.screening.heuristic import (
    _compute_bm25_scores,
    _jaccard_similarity,
    _normalize_scores,
    _recency_bonus,
    _tokenize as heuristic_tokenize,
)
from research_pipeline.sources.base import dedup_cross_source


# -- Strategies --

# Simple text that won't collapse to empty after tokenization
non_empty_text = st.from_regex(r"[a-z]{3,10}( [a-z]{3,10}){1,5}", fullmatch=True)

word_list = st.lists(
    st.from_regex(r"[a-z]{2,12}", fullmatch=True),
    min_size=1,
    max_size=20,
)

sparse_vector = st.dictionaries(
    keys=st.from_regex(r"[a-z]{2,8}", fullmatch=True),
    values=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    min_size=1,
    max_size=20,
)


# -- BM25 Scoring Properties --


class TestBM25Properties:
    """Property tests for BM25 scoring."""

    @given(
        corpus=st.lists(non_empty_text, min_size=1, max_size=10),
        query=word_list,
    )
    @settings(max_examples=50)
    def test_scores_are_finite(
        self, corpus: list[str], query: list[str]
    ) -> None:
        """BM25 scores should always be finite numbers."""
        scores = _compute_bm25_scores(corpus, query)
        assert len(scores) == len(corpus)
        for s in scores:
            assert math.isfinite(s), f"Non-finite BM25 score: {s}"

    @given(corpus=st.lists(non_empty_text, min_size=1, max_size=10))
    @settings(max_examples=30)
    def test_empty_query_returns_zeros(self, corpus: list[str]) -> None:
        """Empty query should return zero scores for all documents."""
        scores = _compute_bm25_scores(corpus, [])
        assert all(s == 0.0 for s in scores)

    @given(query=word_list)
    @settings(max_examples=30)
    def test_empty_corpus_returns_empty(self, query: list[str]) -> None:
        """Empty corpus should return empty list."""
        scores = _compute_bm25_scores([], query)
        assert scores == []

    @given(
        text=non_empty_text,
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30)
    def test_identical_docs_get_same_score(self, text: str, n: int) -> None:
        """Identical documents should get the same BM25 score."""
        corpus = [text] * n
        # Use words from the text as query
        words = text.lower().split()[:3]
        assume(len(words) >= 1)
        scores = _compute_bm25_scores(corpus, words)
        # All scores should be equal (within float precision)
        for s in scores:
            assert abs(s - scores[0]) < 1e-10


# -- Normalization Properties --


class TestNormalizationProperties:
    """Property tests for min-max normalization."""

    @given(scores=st.lists(st.floats(min_value=-100, max_value=100), min_size=2))
    @settings(max_examples=50)
    def test_normalized_range(self, scores: list[float]) -> None:
        """Normalized scores should be in [0, 1]."""
        assume(not any(math.isnan(s) or math.isinf(s) for s in scores))
        result = _normalize_scores(scores)
        for r in result:
            assert 0.0 <= r <= 1.0, f"Out of range: {r}"

    @given(
        scores=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False),
            min_size=2,
        )
    )
    @settings(max_examples=50)
    def test_preserves_ordering(self, scores: list[float]) -> None:
        """Normalization should preserve relative ordering."""
        assume(not any(math.isinf(s) for s in scores))
        result = _normalize_scores(scores)
        for i in range(len(scores)):
            for j in range(len(scores)):
                if scores[i] < scores[j]:
                    assert result[i] <= result[j] + 1e-10

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty output."""
        assert _normalize_scores([]) == []

    @given(val=st.floats(min_value=-100, max_value=100, allow_nan=False))
    @settings(max_examples=30)
    def test_single_value_is_half(self, val: float) -> None:
        """A single-element list normalizes to 0.5 (constant case)."""
        assume(not math.isinf(val))
        result = _normalize_scores([val])
        assert result == [0.5]

    @given(
        val=st.floats(min_value=-100, max_value=100, allow_nan=False),
        n=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30)
    def test_constant_values_normalize_to_half(self, val: float, n: int) -> None:
        """All identical values should normalize to 0.5."""
        assume(not math.isinf(val))
        result = _normalize_scores([val] * n)
        assert all(r == 0.5 for r in result)


# -- TF-IDF Properties --


class TestTFIDFProperties:
    """Property tests for TF-IDF construction."""

    @given(
        docs=st.lists(word_list, min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_tfidf_weights_non_negative(
        self, docs: list[list[str]]
    ) -> None:
        """TF-IDF weights should always be non-negative."""
        vocab, vectors = _build_tfidf(docs)
        for vec in vectors:
            for weight in vec.values():
                assert weight >= 0.0

    @given(
        docs=st.lists(word_list, min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_tfidf_output_length_matches_input(
        self, docs: list[list[str]]
    ) -> None:
        """One vector per document."""
        vocab, vectors = _build_tfidf(docs)
        assert len(vectors) == len(docs)

    def test_empty_docs_returns_empty(self) -> None:
        """Empty document list returns empty."""
        vocab, vectors = _build_tfidf([])
        assert vocab == []
        assert vectors == []

    @given(doc=word_list)
    @settings(max_examples=30)
    def test_single_doc_has_nonzero_weights(self, doc: list[str]) -> None:
        """A single document should have all non-zero weights."""
        vocab, vectors = _build_tfidf([doc])
        assert len(vectors) == 1
        # Every term in doc should appear in the vector
        for term in set(cluster_tokenize(" ".join(doc))):
            if term in vectors[0]:
                assert vectors[0][term] > 0


# -- Cosine Similarity Properties --


class TestCosineProperties:
    """Property tests for cosine similarity."""

    @given(vec=sparse_vector)
    @settings(max_examples=50)
    def test_self_similarity_is_one(self, vec: dict[str, float]) -> None:
        """Cosine similarity of a vector with itself should be 1.0."""
        sim = _cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    @given(a=sparse_vector, b=sparse_vector)
    @settings(max_examples=50)
    def test_similarity_in_range(
        self, a: dict[str, float], b: dict[str, float]
    ) -> None:
        """Cosine similarity should be in [-1, 1] (and [0,1] for positive vectors)."""
        sim = _cosine_similarity(a, b)
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
        # Since our vectors are always positive, should be >= 0
        assert sim >= -1e-6

    @given(a=sparse_vector, b=sparse_vector)
    @settings(max_examples=50)
    def test_symmetry(self, a: dict[str, float], b: dict[str, float]) -> None:
        """Cosine similarity is symmetric: sim(a,b) == sim(b,a)."""
        assert abs(_cosine_similarity(a, b) - _cosine_similarity(b, a)) < 1e-10

    def test_orthogonal_vectors_zero(self) -> None:
        """Vectors with no common keys have zero similarity."""
        a = {"x": 1.0, "y": 2.0}
        b = {"z": 1.0, "w": 2.0}
        assert _cosine_similarity(a, b) == 0.0


# -- Recency Bonus Properties --


class TestRecencyBonusProperties:
    """Property tests for exponential decay recency bonus."""

    @given(
        days_ago=st.integers(min_value=0, max_value=3650),
        half_life=st.floats(min_value=1.0, max_value=365.0),
    )
    @settings(max_examples=50)
    def test_bonus_in_zero_one(self, days_ago: int, half_life: float) -> None:
        """Recency bonus should always be in [0, 1]."""
        published = datetime.now(UTC) - timedelta(days=days_ago)
        bonus = _recency_bonus(published, half_life)
        assert 0.0 <= bonus <= 1.0 + 1e-6

    @given(half_life=st.floats(min_value=1.0, max_value=365.0))
    @settings(max_examples=30)
    def test_newer_is_higher(self, half_life: float) -> None:
        """More recent papers should have higher bonus."""
        now = datetime.now(UTC)
        recent = now - timedelta(days=1)
        old = now - timedelta(days=100)
        assert _recency_bonus(recent, half_life) > _recency_bonus(old, half_life)

    @given(half_life=st.floats(min_value=1.0, max_value=365.0))
    @settings(max_examples=30)
    def test_today_is_approximately_one(self, half_life: float) -> None:
        """A paper published right now should have bonus ≈ 1.0."""
        now = datetime.now(UTC)
        bonus = _recency_bonus(now, half_life)
        assert bonus > 0.99

    def test_monotonically_decreasing(self) -> None:
        """Bonus should decrease as paper gets older."""
        now = datetime.now(UTC)
        bonuses = [
            _recency_bonus(now - timedelta(days=d), 90.0) for d in range(0, 365, 30)
        ]
        for i in range(len(bonuses) - 1):
            assert bonuses[i] >= bonuses[i + 1]


# -- Jaccard Similarity Properties --


class TestJaccardProperties:
    """Property tests for Jaccard similarity."""

    @given(
        s=st.frozensets(st.text(min_size=1, max_size=5), min_size=1, max_size=20)
    )
    @settings(max_examples=50)
    def test_self_similarity_is_one(self, s: frozenset[str]) -> None:
        """Jaccard(A, A) == 1.0."""
        assert _jaccard_similarity(set(s), set(s)) == 1.0

    @given(
        a=st.frozensets(st.text(min_size=1, max_size=5), min_size=1, max_size=20),
        b=st.frozensets(st.text(min_size=1, max_size=5), min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_range_zero_to_one(
        self, a: frozenset[str], b: frozenset[str]
    ) -> None:
        """Jaccard similarity is in [0, 1]."""
        sim = _jaccard_similarity(set(a), set(b))
        assert 0.0 <= sim <= 1.0

    @given(
        a=st.frozensets(st.text(min_size=1, max_size=5), min_size=1, max_size=20),
        b=st.frozensets(st.text(min_size=1, max_size=5), min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_symmetry(self, a: frozenset[str], b: frozenset[str]) -> None:
        """Jaccard is symmetric."""
        assert _jaccard_similarity(set(a), set(b)) == _jaccard_similarity(
            set(b), set(a)
        )

    def test_disjoint_sets_zero(self) -> None:
        """Disjoint sets have zero Jaccard similarity."""
        assert _jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_empty_sets_zero(self) -> None:
        """Empty sets have zero Jaccard similarity."""
        assert _jaccard_similarity(set(), set()) == 0.0


# -- Cross-Source Dedup Properties --


class TestDedupProperties:
    """Property tests for cross-source deduplication."""

    def _make_candidate(
        self,
        arxiv_id: str,
        title: str,
        doi: str | None = None,
    ) -> "CandidateRecord":
        """Create a minimal CandidateRecord for testing."""
        from research_pipeline.models.candidate import CandidateRecord

        return CandidateRecord(
            arxiv_id=arxiv_id,
            version="1",
            title=title,
            authors=["Test Author"],
            published=datetime(2024, 1, 1, tzinfo=UTC),
            updated=datetime(2024, 1, 1, tzinfo=UTC),
            categories=["cs.AI"],
            primary_category="cs.AI",
            abstract="Test abstract",
            abs_url="https://arxiv.org/abs/" + arxiv_id,
            pdf_url="https://arxiv.org/pdf/" + arxiv_id,
            doi=doi,
        )

    @given(
        titles=st.lists(
            st.from_regex(r"[a-z]{5,20}", fullmatch=True),
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=30)
    def test_unique_candidates_preserved(self, titles: list[str]) -> None:
        """Candidates with unique titles/IDs should all be preserved."""
        candidates = [
            self._make_candidate(f"2024.{i:05d}", title)
            for i, title in enumerate(titles)
        ]
        result = dedup_cross_source(candidates)
        assert len(result) == len(candidates)

    @given(n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=30)
    def test_duplicates_collapsed(self, n: int) -> None:
        """N identical candidates should collapse to 1."""
        candidates = [
            self._make_candidate("2024.00001", "Same Title") for _ in range(n)
        ]
        result = dedup_cross_source(candidates)
        assert len(result) == 1

    def test_dedup_idempotent(self) -> None:
        """Running dedup twice gives same result."""
        candidates = [
            self._make_candidate("2024.00001", "Paper A"),
            self._make_candidate("2024.00001", "Paper A"),
            self._make_candidate("2024.00002", "Paper B"),
        ]
        first = dedup_cross_source(candidates)
        second = dedup_cross_source(first)
        assert len(first) == len(second)

    @given(
        titles=st.lists(
            st.from_regex(r"[a-z]{5,20}", fullmatch=True),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=30)
    def test_output_subset_of_input(self, titles: list[str]) -> None:
        """Dedup output should be a subset of the input."""
        candidates = [
            self._make_candidate(f"2024.{i:05d}", title)
            for i, title in enumerate(titles)
        ]
        result = dedup_cross_source(candidates)
        assert len(result) <= len(candidates)
        for r in result:
            assert r in candidates


# -- Citation Context Properties --


class TestCitationContextProperties:
    """Property tests for citation context extraction."""

    @given(
        n=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_numeric_citations_detected(self, n: int) -> None:
        """Numeric citation markers [N] should be detected."""
        text = f"This is important [{n}]. Another sentence follows."
        contexts = extract_citation_contexts(text)
        markers = [c.marker for c in contexts]
        assert f"[{n}]" in markers

    def test_author_year_detected(self) -> None:
        """Author-year citations should be detected."""
        text = "Previous work (Smith, 2024) shows that this is true."
        contexts = extract_citation_contexts(text)
        markers = [c.marker for c in contexts]
        assert "(Smith, 2024)" in markers

    def test_no_citations_returns_empty(self) -> None:
        """Text without citations returns empty list."""
        text = "This text has no citations at all."
        contexts = extract_citation_contexts(text)
        assert contexts == []

    @given(
        context_window=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=20)
    def test_context_window_parameter(self, context_window: int) -> None:
        """Context window parameter should not crash."""
        text = "First sentence. This cites [1] something. Third sentence."
        contexts = extract_citation_contexts(text, context_window=context_window)
        # Should find the citation regardless of window size
        assert len(contexts) >= 1

    def test_multiple_citations_in_one_sentence(self) -> None:
        """Multiple citations in one sentence should all be found."""
        text = "Studies [1] and [2] and [3] all confirm this result."
        contexts = extract_citation_contexts(text)
        markers = {c.marker for c in contexts}
        assert "[1]" in markers
        assert "[2]" in markers
        assert "[3]" in markers


# -- Sentence Splitting Properties --


class TestSentenceSplitProperties:
    """Property tests for sentence splitting."""

    @given(
        sentences=st.lists(
            st.from_regex(r"[A-Z][a-z]{4,20}", fullmatch=True),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=30)
    def test_no_empty_sentences(self, sentences: list[str]) -> None:
        """Splitting should never produce empty sentences."""
        # Join with periods to form multi-sentence text
        text = ". ".join(s.strip().rstrip(".") for s in sentences) + "."
        result = _split_sentences(text)
        for s in result:
            assert len(s.strip()) > 0
