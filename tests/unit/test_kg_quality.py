"""Tests for KG quality evaluation framework."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from research_pipeline.quality.kg_quality import (
    ConsistencyMetrics,
    StructuralMetrics,
    _accuracy_from_structural,
    _jaccard,
    _name_tokens,
    compute_completeness_metrics,
    compute_consistency_metrics,
    compute_redundancy_metrics,
    compute_structural_metrics,
    compute_timeliness_metrics,
    evaluate_kg_quality,
    sample_triples_twcs,
)

# ---------------------------------------------------------------------------
# Fixture: in-memory KG database
# ---------------------------------------------------------------------------


def _create_kg_schema(conn: sqlite3.Connection) -> None:
    """Create the KG tables matching storage/knowledge_graph.py schema."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            properties TEXT DEFAULT '{}'
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS triples (
            subject_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            object_id TEXT NOT NULL,
            provenance_paper TEXT DEFAULT '',
            provenance_run TEXT DEFAULT '',
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT ''
        )"""
    )


def _make_kg(
    entities: list[tuple[str, str, str, str]] | None = None,
    triples: list[tuple[str, str, str, str, str, float, str]] | None = None,
) -> sqlite3.Connection:
    """Create an in-memory KG with given entities and triples."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_kg_schema(conn)

    for e in entities or []:
        conn.execute(
            "INSERT INTO entities (entity_id, entity_type, name, properties)"
            " VALUES (?, ?, ?, ?)",
            e,
        )
    for t in triples or []:
        conn.execute(
            "INSERT INTO triples"
            " (subject_id, relation, object_id,"
            "  provenance_paper, provenance_run, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            t,
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Empty KG
# ---------------------------------------------------------------------------


class TestEmptyKG:
    """Tests on an empty knowledge graph."""

    def test_structural_empty(self) -> None:
        conn = _make_kg()
        m = compute_structural_metrics(conn)
        assert m.num_entities == 0
        assert m.num_triples == 0
        assert m.icr == 0.0

    def test_consistency_empty(self) -> None:
        conn = _make_kg()
        m = compute_consistency_metrics(conn)
        assert m.ic_score == 1.0
        assert m.ec_score == 0.0

    def test_completeness_empty(self) -> None:
        conn = _make_kg()
        m = compute_completeness_metrics(conn)
        assert m.entity_type_coverage == 0.0
        assert m.orphan_entities == 0

    def test_timeliness_empty(self) -> None:
        conn = _make_kg()
        m = compute_timeliness_metrics(conn)
        assert m.avg_age_days == 0.0

    def test_redundancy_empty(self) -> None:
        conn = _make_kg()
        m = compute_redundancy_metrics(conn)
        assert m.exact_duplicate_triples == 0

    def test_composite_empty(self) -> None:
        conn = _make_kg()
        score = evaluate_kg_quality(conn)
        # Empty KG: timeliness=1.0 (no stale), redundancy=1.0 (no dups),
        # accuracy has partial score from no contradictions/dups.
        # Completeness=0.0 (no types). Composite is a weighted average.
        assert score.completeness == 0.0
        assert score.timeliness == 1.0
        assert score.redundancy == 1.0


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------


class TestStructuralMetrics:
    """Tests for Layer 1 structural metrics."""

    def test_icr_all_connected(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
        ]
        triples = [("p1", "addresses_concept", "c1", "", "", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_structural_metrics(conn)
        assert m.icr == 1.0

    def test_icr_partial_connected(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
            ("m1", "method", "Method 1", "{}"),
        ]
        triples = [("p1", "addresses_concept", "c1", "", "", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_structural_metrics(conn)
        assert abs(m.icr - 2 / 3) < 0.01

    def test_density(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
        ]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, ""),
            ("p1", "uses_method", "c1", "", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_structural_metrics(conn)
        assert m.density == 1.0  # 2 triples / 2 entities

    def test_connected_components(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
            ("p2", "paper", "Paper 2", "{}"),
            ("c2", "concept", "Concept 2", "{}"),
        ]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, ""),
            ("p2", "addresses_concept", "c2", "", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_structural_metrics(conn)
        assert m.connected_components == 2
        assert m.largest_component_fraction == 0.5

    def test_type_distribution(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("p2", "paper", "Paper 2", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
        ]
        conn = _make_kg(entities)
        m = compute_structural_metrics(conn)
        assert m.type_distribution["paper"] == 2
        assert m.type_distribution["concept"] == 1

    def test_avg_degree(self) -> None:
        entities = [
            ("p1", "paper", "Paper 1", "{}"),
            ("c1", "concept", "Concept 1", "{}"),
        ]
        triples = [("p1", "addresses_concept", "c1", "", "", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_structural_metrics(conn)
        # Undirected: p1↔c1, each has 1 neighbor, degree_sum=2, avg=2/2=1.0
        assert m.avg_degree == 1.0


# ---------------------------------------------------------------------------
# Consistency metrics
# ---------------------------------------------------------------------------


class TestConsistencyMetrics:
    """Tests for Layer 2 consistency metrics."""

    def test_no_contradictions(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "claim", "C1", "{}")]
        triples = [("p1", "supports_claim", "c1", "ref1", "run1", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_consistency_metrics(conn)
        assert m.ic_contradiction_count == 0
        assert m.ic_score == 1.0

    def test_contradiction_detected(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "claim", "C1", "{}")]
        triples = [
            ("p1", "supports_claim", "c1", "ref1", "run1", 1.0, ""),
            ("p1", "contradicts_claim", "c1", "ref2", "run1", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_consistency_metrics(conn)
        assert m.ic_contradiction_count >= 1
        assert m.ic_score < 1.0

    def test_provenance_coverage(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [
            ("p1", "addresses_concept", "c1", "ref1", "", 1.0, ""),
            ("p1", "uses_method", "c1", "", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_consistency_metrics(conn)
        assert m.ec_provenance_coverage == 0.5

    def test_duplicate_detection(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, ""),
            ("p1", "addresses_concept", "c1", "", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_consistency_metrics(conn)
        assert m.duplicate_triples == 1


# ---------------------------------------------------------------------------
# Completeness metrics
# ---------------------------------------------------------------------------


class TestCompletenessMetrics:
    """Tests for completeness evaluation."""

    def test_full_type_coverage(self) -> None:
        entities = [
            ("p1", "paper", "P", "{}"),
            ("c1", "concept", "C", "{}"),
        ]
        conn = _make_kg(entities)
        m = compute_completeness_metrics(
            conn, expected_entity_types=["paper", "concept"]
        )
        assert m.entity_type_coverage == 1.0
        assert len(m.missing_entity_types) == 0

    def test_partial_type_coverage(self) -> None:
        entities = [("p1", "paper", "P", "{}")]
        conn = _make_kg(entities)
        m = compute_completeness_metrics(
            conn, expected_entity_types=["paper", "concept", "method"]
        )
        assert abs(m.entity_type_coverage - 1 / 3) < 0.01
        assert "concept" in m.missing_entity_types

    def test_orphan_entities(self) -> None:
        entities = [
            ("p1", "paper", "P1", "{}"),
            ("c1", "concept", "C1", "{}"),
            ("m1", "method", "M1", "{}"),
        ]
        triples = [("p1", "addresses_concept", "c1", "", "", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_completeness_metrics(conn)
        assert m.orphan_entities == 1  # m1 has no edges

    def test_avg_properties(self) -> None:
        entities = [
            ("p1", "paper", "P1", '{"year": "2024", "doi": "10.1234"}'),
            ("c1", "concept", "C1", "{}"),
        ]
        conn = _make_kg(entities)
        m = compute_completeness_metrics(conn)
        assert m.avg_properties_per_entity == 1.0  # (2 + 0) / 2


# ---------------------------------------------------------------------------
# Timeliness metrics
# ---------------------------------------------------------------------------


class TestTimelinessMetrics:
    """Tests for temporal freshness."""

    def test_fresh_triples(self) -> None:
        now = datetime(2024, 6, 1, tzinfo=UTC)
        yesterday = (now - timedelta(days=1)).isoformat()
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, yesterday),
        ]
        conn = _make_kg(entities, triples)
        m = compute_timeliness_metrics(conn, staleness_days=365, now=now)
        assert m.avg_age_days < 2.0
        assert m.staleness_ratio == 0.0

    def test_stale_triples(self) -> None:
        now = datetime(2024, 6, 1, tzinfo=UTC)
        old = (now - timedelta(days=400)).isoformat()
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, old),
        ]
        conn = _make_kg(entities, triples)
        m = compute_timeliness_metrics(conn, staleness_days=365, now=now)
        assert m.staleness_ratio == 1.0
        assert m.avg_age_days > 390

    def test_mixed_freshness(self) -> None:
        now = datetime(2024, 6, 1, tzinfo=UTC)
        fresh = (now - timedelta(days=10)).isoformat()
        stale = (now - timedelta(days=500)).isoformat()
        entities = [
            ("p1", "paper", "P1", "{}"),
            ("c1", "concept", "C1", "{}"),
        ]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, fresh),
            ("p1", "uses_method", "c1", "", "", 1.0, stale),
        ]
        conn = _make_kg(entities, triples)
        m = compute_timeliness_metrics(conn, staleness_days=365, now=now)
        assert m.staleness_ratio == 0.5
        assert m.newest_triple_age_days < 15


# ---------------------------------------------------------------------------
# Redundancy metrics
# ---------------------------------------------------------------------------


class TestRedundancyMetrics:
    """Tests for redundancy detection."""

    def test_no_redundancy(self) -> None:
        entities = [
            ("p1", "paper", "Transformer Attention", "{}"),
            ("p2", "paper", "Graph Neural Networks", "{}"),
        ]
        triples = [("p1", "cites", "p2", "", "", 1.0, "")]
        conn = _make_kg(entities, triples)
        m = compute_redundancy_metrics(conn)
        assert m.exact_duplicate_triples == 0
        assert m.near_duplicate_entities == 0

    def test_duplicate_triples(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("p2", "paper", "P2", "{}")]
        triples = [
            ("p1", "cites", "p2", "", "", 1.0, ""),
            ("p1", "cites", "p2", "", "", 1.0, ""),
            ("p1", "cites", "p2", "", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        m = compute_redundancy_metrics(conn)
        assert m.exact_duplicate_triples == 2

    def test_near_duplicate_entities(self) -> None:
        entities = [
            ("p1", "paper", "Attention Is All You Need", "{}"),
            ("p2", "paper", "Attention Is All You Need 2024", "{}"),
        ]
        conn = _make_kg(entities)
        m = compute_redundancy_metrics(conn, similarity_threshold=0.7)
        assert m.near_duplicate_entities >= 1


# ---------------------------------------------------------------------------
# TWCS Sampling
# ---------------------------------------------------------------------------


class TestTWCSSampling:
    """Tests for Type-Weighted Cluster Sampling."""

    def test_empty_kg_sample(self) -> None:
        conn = _make_kg()
        sample = sample_triples_twcs(conn, sample_size=10)
        assert sample == []

    def test_sample_preserves_types(self) -> None:
        entities = [
            ("p1", "paper", "P1", "{}"),
            ("c1", "concept", "C1", "{}"),
            ("m1", "method", "M1", "{}"),
        ]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, ""),
            ("p1", "addresses_concept", "c1", "r", "", 1.0, ""),
            ("p1", "addresses_concept", "c1", "r2", "", 1.0, ""),
            ("p1", "uses_method", "m1", "", "", 1.0, ""),
            ("p1", "uses_method", "m1", "r", "", 1.0, ""),
        ]
        conn = _make_kg(entities, triples)
        sample = sample_triples_twcs(conn, sample_size=4, seed=42)
        rels = {s["relation"] for s in sample}
        assert "addresses_concept" in rels
        assert "uses_method" in rels

    def test_sample_size_limit(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [
            ("p1", "addresses_concept", "c1", "", "", 1.0, "") for _ in range(20)
        ]
        conn = _make_kg(entities, triples)
        sample = sample_triples_twcs(conn, sample_size=5, seed=42)
        assert len(sample) <= 5

    def test_sample_reproducible(self) -> None:
        entities = [("p1", "paper", "P1", "{}"), ("c1", "concept", "C1", "{}")]
        triples = [("p1", "cites", "c1", f"r{i}", "", 1.0, "") for i in range(10)]
        conn = _make_kg(entities, triples)
        s1 = sample_triples_twcs(conn, sample_size=3, seed=99)
        s2 = sample_triples_twcs(conn, sample_size=3, seed=99)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------


class TestCompositeEvaluation:
    """Tests for the full evaluate_kg_quality pipeline."""

    def test_perfect_kg(self) -> None:
        now = datetime(2024, 6, 1, tzinfo=UTC)
        recent = (now - timedelta(days=5)).isoformat()
        entities = [
            ("p1", "paper", "P1", '{"year": "2024"}'),
            ("c1", "concept", "C1", '{"domain": "ML"}'),
            ("m1", "method", "M1", '{"type": "DL"}'),
            ("e1", "experiment", "E1", "{}"),
            ("cl1", "claim", "CL1", "{}"),
            ("a1", "author", "A1", "{}"),
            ("v1", "venue", "V1", "{}"),
        ]
        triples = [
            ("p1", "addresses_concept", "c1", "ref", "run1", 1.0, recent),
            ("p1", "uses_method", "m1", "ref", "run1", 1.0, recent),
            ("p1", "evaluates_on", "e1", "ref", "run1", 1.0, recent),
            ("p1", "supports_claim", "cl1", "ref", "run1", 1.0, recent),
            ("p1", "authored_by", "a1", "ref", "run1", 1.0, recent),
            ("p1", "published_in", "v1", "ref", "run1", 1.0, recent),
            ("p1", "cites", "p1", "ref", "run1", 1.0, recent),
            ("p1", "proposes_method", "m1", "ref", "run1", 1.0, recent),
            ("p1", "related_to", "c1", "ref", "run1", 1.0, recent),
            ("p1", "contradicts_claim", "cl1", "ref2", "run1", 1.0, recent),
        ]
        conn = _make_kg(entities, triples)
        score = evaluate_kg_quality(conn, now=now)
        assert score.composite > 0.5
        assert score.timeliness > 0.9
        assert score.consistency > 0.0

    def test_poor_kg(self) -> None:
        now = datetime(2024, 6, 1, tzinfo=UTC)
        old = (now - timedelta(days=1000)).isoformat()
        entities = [
            ("p1", "paper", "P1", "{}"),
        ]
        triples = [
            ("p1", "cites", "p1", "", "", 0.5, old),
            ("p1", "cites", "p1", "", "", 0.5, old),
        ]
        conn = _make_kg(entities, triples)
        score = evaluate_kg_quality(conn, now=now)
        assert score.composite < 0.5
        assert score.timeliness < 0.5

    def test_custom_weights(self) -> None:
        entities = [
            ("p1", "paper", "P1", "{}"),
            ("c1", "concept", "C1", "{}"),
        ]
        triples = [("p1", "addresses_concept", "c1", "ref", "run", 1.0, "")]
        conn = _make_kg(entities, triples)
        w = {
            "accuracy": 1.0,
            "consistency": 0.0,
            "completeness": 0.0,
            "timeliness": 0.0,
            "redundancy": 0.0,
        }
        score = evaluate_kg_quality(conn, weights=w)
        assert score.composite >= 0.0

    def test_to_dict(self) -> None:
        conn = _make_kg()
        score = evaluate_kg_quality(conn)
        d = score.to_dict()
        assert "composite" in d
        assert "structural" in d
        assert "consistency_detail" in d


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_name_tokens(self) -> None:
        assert _name_tokens("Hello World") == {"hello", "world"}
        assert _name_tokens("") == set()  # "".split() → []

    def test_jaccard_identical(self) -> None:
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self) -> None:
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self) -> None:
        assert abs(_jaccard({"a", "b", "c"}, {"a", "b", "d"}) - 0.5) < 0.01

    def test_jaccard_empty(self) -> None:
        assert _jaccard(set(), set()) == 1.0

    def test_accuracy_from_structural(self) -> None:
        struct = StructuralMetrics(num_entities=10, num_triples=20, icr=1.0)
        consist = ConsistencyMetrics(ic_score=1.0, duplicate_triples=0)
        acc = _accuracy_from_structural(struct, consist)
        assert acc > 0.7
