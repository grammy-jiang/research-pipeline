"""Tests for storage.knowledge_graph module."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from research_pipeline.storage.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    RelationType,
    Triple,
)

# ── Entity CRUD ──────────────────────────────────────────────────────


class TestEntityCRUD:
    """Entity add / get / upsert / list / count."""

    def test_add_and_get_entity(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper One")
            entity = kg.get_entity("p1")
            assert entity is not None
            assert entity.entity_id == "p1"
            assert entity.entity_type == EntityType.PAPER
            assert entity.name == "Paper One"
            assert entity.properties == {}
        finally:
            kg.close()

    def test_add_entity_with_properties(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity(
                "p2",
                EntityType.PAPER,
                "Paper Two",
                properties={"doi": "10.1234/test"},
            )
            entity = kg.get_entity("p2")
            assert entity is not None
            assert entity.properties == {"doi": "10.1234/test"}
        finally:
            kg.close()

    def test_upsert_entity(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Old Name")
            kg.add_entity("p1", EntityType.PAPER, "New Name", {"key": "val"})
            entity = kg.get_entity("p1")
            assert entity is not None
            assert entity.name == "New Name"
            assert entity.properties == {"key": "val"}
        finally:
            kg.close()

    def test_get_entity_not_found(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            assert kg.get_entity("nonexistent") is None
        finally:
            kg.close()

    def test_get_entities_by_type(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("p2", EntityType.PAPER, "Paper 2")
            kg.add_entity("m1", EntityType.METHOD, "Method 1")

            papers = kg.get_entities_by_type(EntityType.PAPER)
            assert len(papers) == 2
            methods = kg.get_entities_by_type(EntityType.METHOD)
            assert len(methods) == 1
            concepts = kg.get_entities_by_type(EntityType.CONCEPT)
            assert len(concepts) == 0
        finally:
            kg.close()

    def test_count_entities(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("a1", EntityType.AUTHOR, "Author 1")
            counts = kg.count_entities()
            assert counts["paper"] == 1
            assert counts["author"] == 1
        finally:
            kg.close()


# ── Triple CRUD ──────────────────────────────────────────────────────


class TestTripleCRUD:
    """Triple add / get / filter / neighbors / count."""

    def test_add_and_get_triple(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("p2", EntityType.PAPER, "Paper 2")
            kg.add_triple("p1", RelationType.CITES, "p2")

            triples = kg.get_triples(subject_id="p1")
            assert len(triples) == 1
            assert triples[0].relation == RelationType.CITES
            assert triples[0].object_id == "p2"
        finally:
            kg.close()

    def test_triple_with_provenance(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("m1", EntityType.METHOD, "BERT")
            kg.add_triple(
                "p1",
                RelationType.USES_METHOD,
                "m1",
                provenance_paper="p1",
                provenance_run="run-001",
                confidence=0.85,
            )
            triples = kg.get_triples(provenance_paper="p1")
            assert len(triples) == 1
            assert triples[0].provenance_run == "run-001"
            assert triples[0].confidence == pytest.approx(0.85)
        finally:
            kg.close()

    def test_duplicate_triple_ignored(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("p2", EntityType.PAPER, "Paper 2")
            kg.add_triple("p1", RelationType.CITES, "p2", provenance_paper="p1")
            kg.add_triple("p1", RelationType.CITES, "p2", provenance_paper="p1")
            triples = kg.get_triples(subject_id="p1")
            assert len(triples) == 1
        finally:
            kg.close()

    def test_get_triples_with_relation_filter(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("p2", EntityType.PAPER, "Paper 2")
            kg.add_entity("m1", EntityType.METHOD, "Method 1")
            kg.add_triple("p1", RelationType.CITES, "p2")
            kg.add_triple("p1", RelationType.USES_METHOD, "m1")

            cites = kg.get_triples(relation=RelationType.CITES)
            assert len(cites) == 1
            uses = kg.get_triples(relation=RelationType.USES_METHOD)
            assert len(uses) == 1
        finally:
            kg.close()

    def test_get_triples_with_object_filter(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "Paper 1")
            kg.add_entity("p2", EntityType.PAPER, "Paper 2")
            kg.add_entity("p3", EntityType.PAPER, "Paper 3")
            kg.add_triple("p1", RelationType.CITES, "p2")
            kg.add_triple("p3", RelationType.CITES, "p2")

            triples = kg.get_triples(object_id="p2")
            assert len(triples) == 2
        finally:
            kg.close()

    def test_get_triples_no_filter(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_triple("p1", RelationType.CITES, "p2")
            triples = kg.get_triples()
            assert len(triples) == 1
        finally:
            kg.close()

    def test_get_neighbors_outgoing(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_entity("m1", EntityType.METHOD, "M1")
            kg.add_triple("p1", RelationType.CITES, "p2")
            kg.add_triple("p1", RelationType.USES_METHOD, "m1")

            out = kg.get_neighbors("p1", direction="outgoing")
            assert len(out) == 2
        finally:
            kg.close()

    def test_get_neighbors_incoming(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_triple("p1", RelationType.CITES, "p2")

            inc = kg.get_neighbors("p2", direction="incoming")
            assert len(inc) == 1
            assert inc[0].subject_id == "p1"
        finally:
            kg.close()

    def test_get_neighbors_both(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_entity("p3", EntityType.PAPER, "P3")
            kg.add_triple("p1", RelationType.CITES, "p2")
            kg.add_triple("p3", RelationType.CITES, "p2")

            both = kg.get_neighbors("p2", direction="both")
            # p2 is object in both triples (incoming)
            assert len(both) == 2
        finally:
            kg.close()

    def test_count_triples(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_entity("m1", EntityType.METHOD, "M1")
            kg.add_triple("p1", RelationType.CITES, "p2")
            kg.add_triple("p1", RelationType.USES_METHOD, "m1")

            counts = kg.count_triples()
            assert counts["cites"] == 1
            assert counts["uses_method"] == 1
        finally:
            kg.close()


# ── Stats ────────────────────────────────────────────────────────────


class TestStats:
    """Summary statistics."""

    def test_stats_empty(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            s = kg.stats()
            assert s["total_entities"] == 0
            assert s["total_triples"] == 0
            assert s["entities"] == {}
            assert s["triples"] == {}
        finally:
            kg.close()

    def test_stats_populated(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("a1", EntityType.AUTHOR, "Auth")
            kg.add_triple("p1", RelationType.AUTHORED_BY, "a1")

            s = kg.stats()
            assert s["total_entities"] == 2
            assert s["total_triples"] == 1
        finally:
            kg.close()


# ── Clear / Close ────────────────────────────────────────────────────


class TestClearClose:
    """clear() and close() behavior."""

    def test_clear(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            kg.add_entity("p1", EntityType.PAPER, "P1")
            kg.add_entity("p2", EntityType.PAPER, "P2")
            kg.add_triple("p1", RelationType.CITES, "p2")

            kg.clear()
            assert kg.stats()["total_entities"] == 0
            assert kg.stats()["total_triples"] == 0
        finally:
            kg.close()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "kg.db"
        kg = KnowledgeGraph(db_path=db)
        kg.add_entity("p1", EntityType.PAPER, "Paper 1")
        kg.close()
        assert kg._conn is None

        # Reopen
        kg2 = KnowledgeGraph(db_path=db)
        try:
            entity = kg2.get_entity("p1")
            assert entity is not None
            assert entity.name == "Paper 1"
        finally:
            kg2.close()


# ── Ingestion ────────────────────────────────────────────────────────


class TestIngestFromCandidates:
    """ingest_from_candidates() with mock CandidateRecord objects."""

    def test_ingest_papers(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            candidate = SimpleNamespace(
                arxiv_id="2401.12345",
                title="Test Paper",
                doi="10.1234/test",
                published="2024-01-15",
                authors=["Alice Smith", "Bob Jones"],
            )
            count = kg.ingest_from_candidates([candidate], run_id="run-001")
            assert count == 1

            paper = kg.get_entity("2401.12345")
            assert paper is not None
            assert paper.name == "Test Paper"
            assert paper.properties["doi"] == "10.1234/test"

            # Authors should be ingested
            authors = kg.get_entities_by_type(EntityType.AUTHOR)
            assert len(authors) == 2

            # AUTHORED_BY triples
            triples = kg.get_triples(
                subject_id="2401.12345",
                relation=RelationType.AUTHORED_BY,
            )
            assert len(triples) == 2
        finally:
            kg.close()

    def test_ingest_skips_no_id(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            candidate = SimpleNamespace(title="No ID")
            count = kg.ingest_from_candidates([candidate])
            assert count == 0
        finally:
            kg.close()

    def test_ingest_paper_id_fallback(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            candidate = SimpleNamespace(
                paper_id="s2-abc123",
                title="S2 Paper",
                authors=[],
            )
            count = kg.ingest_from_candidates([candidate])
            assert count == 1
            assert kg.get_entity("s2-abc123") is not None
        finally:
            kg.close()

    def test_ingest_no_authors(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            candidate = SimpleNamespace(
                arxiv_id="2401.00001",
                title="Solo Paper",
                authors=None,
            )
            count = kg.ingest_from_candidates([candidate])
            assert count == 1
            triples = kg.get_triples(subject_id="2401.00001")
            assert len(triples) == 0
        finally:
            kg.close()


class TestIngestFromClaims:
    """ingest_from_claims() with mock ClaimDecomposition objects."""

    @staticmethod
    def _make_decomp(evidence_class_value: str = "supported") -> SimpleNamespace:
        """Build a mock ClaimDecomposition."""
        evidence_class = SimpleNamespace(value=evidence_class_value)
        claim = SimpleNamespace(
            claim_id="CL-001",
            statement="Transformers improve accuracy by 10%",
            source_type="finding",
            evidence_class=evidence_class,
            confidence_score=0.9,
        )
        return SimpleNamespace(
            paper_id="2401.12345",
            title="Test Paper",
            claims=[claim],
        )

    def test_ingest_claims_supported(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            decomp = self._make_decomp("supported")
            count = kg.ingest_from_claims(decomp, run_id="run-001")
            assert count == 1

            claim_entity = kg.get_entity("claim:2401.12345:CL-001")
            assert claim_entity is not None
            assert claim_entity.entity_type == EntityType.CLAIM
            assert claim_entity.properties["evidence_class"] == "supported"

            triples = kg.get_triples(subject_id="2401.12345")
            assert len(triples) == 1
            assert triples[0].relation == RelationType.SUPPORTS_CLAIM
            assert triples[0].confidence == pytest.approx(0.9)
        finally:
            kg.close()

    def test_ingest_claims_conflicting(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            decomp = self._make_decomp("conflicting")
            count = kg.ingest_from_claims(decomp, run_id="run-002")
            assert count == 1

            triples = kg.get_triples(subject_id="2401.12345")
            assert triples[0].relation == RelationType.CONTRADICTS_CLAIM
        finally:
            kg.close()

    def test_ingest_claims_creates_paper(self, tmp_path: Path) -> None:
        kg = KnowledgeGraph(db_path=tmp_path / "kg.db")
        try:
            decomp = self._make_decomp()
            kg.ingest_from_claims(decomp)
            paper = kg.get_entity("2401.12345")
            assert paper is not None
            assert paper.entity_type == EntityType.PAPER
        finally:
            kg.close()


# ── Model roundtrip tests ────────────────────────────────────────────


class TestModels:
    """Pydantic model serialization roundtrips."""

    def test_entity_roundtrip(self) -> None:
        e = Entity(
            entity_id="p1",
            entity_type=EntityType.PAPER,
            name="Test",
            properties={"doi": "10.1234"},
        )
        data = e.model_dump()
        e2 = Entity.model_validate(data)
        assert e == e2

    def test_triple_roundtrip(self) -> None:
        t = Triple(
            subject_id="p1",
            relation=RelationType.CITES,
            object_id="p2",
            provenance_paper="p1",
            provenance_run="run-001",
            confidence=0.95,
            created_at="2024-01-01T00:00:00+00:00",
        )
        data = t.model_dump()
        t2 = Triple.model_validate(data)
        assert t == t2

    def test_entity_type_values(self) -> None:
        expected = {
            "paper",
            "concept",
            "method",
            "experiment",
            "claim",
            "author",
            "venue",
        }
        assert {e.value for e in EntityType} == expected

    def test_relation_type_values(self) -> None:
        expected = {
            "cites",
            "uses_method",
            "proposes_method",
            "addresses_concept",
            "evaluates_on",
            "related_to",
            "supports_claim",
            "contradicts_claim",
            "authored_by",
            "published_in",
        }
        assert {r.value for r in RelationType} == expected


# ── CLI handlers ─────────────────────────────────────────────────────


class TestCLIHandlers:
    """CLI handler functions with mock data."""

    def test_run_kg_stats(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_stats

        db = tmp_path / "kg.db"
        kg = KnowledgeGraph(db_path=db)
        kg.add_entity("p1", EntityType.PAPER, "P1")
        kg.close()

        run_kg_stats(db_path=db)
        captured = capsys.readouterr()
        assert "Total entities: 1" in captured.out

    def test_run_kg_query_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_query

        db = tmp_path / "kg.db"
        kg = KnowledgeGraph(db_path=db)
        kg.add_entity("p1", EntityType.PAPER, "Test Paper")
        kg.add_entity("a1", EntityType.AUTHOR, "Alice")
        kg.add_triple("p1", RelationType.AUTHORED_BY, "a1")
        kg.close()

        run_kg_query("p1", db_path=db)
        captured = capsys.readouterr()
        assert "Test Paper" in captured.out
        assert "authored_by" in captured.out

    def test_run_kg_query_not_found(self, tmp_path: Path) -> None:
        from click.exceptions import Exit

        from research_pipeline.cli.cmd_kg import run_kg_query

        db = tmp_path / "kg.db"
        KnowledgeGraph(db_path=db).close()

        with pytest.raises(Exit):
            run_kg_query("nonexistent", db_path=db)

    def test_run_kg_stats_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from research_pipeline.cli.cmd_kg import run_kg_stats

        db = tmp_path / "kg.db"
        KnowledgeGraph(db_path=db).close()

        run_kg_stats(db_path=db)
        captured = capsys.readouterr()
        assert "Total entities: 0" in captured.out
        assert "Total triples: 0" in captured.out
