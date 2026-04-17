"""Tests for Case-Based Reasoning (CBR) module (C1)."""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.memory.cbr import (
    Adaptation,
    Case,
    CaseMatch,
    CaseStore,
    StrategyRecommendation,
    cbr_lookup,
    cbr_retain,
    create_case_from_run,
    recommend_strategy,
    record_adaptation,
    retrieve_similar_cases,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestCase:
    """Tests for Case dataclass."""

    def test_creation(self) -> None:
        c = Case(case_id="run-1", topic="transformer architectures")
        assert c.case_id == "run-1"
        assert c.topic == "transformer architectures"
        assert c.outcome == "unknown"

    def test_defaults(self) -> None:
        c = Case(case_id="r", topic="t")
        assert c.query_terms == []
        assert c.sources_used == []
        assert c.synthesis_quality == 0.0
        assert c.pipeline_profile == "standard"

    def test_to_dict(self) -> None:
        c = Case(case_id="r1", topic="llm agents", sources_used=["arxiv", "scholar"])
        d = c.to_dict()
        assert d["case_id"] == "r1"
        assert d["sources_used"] == ["arxiv", "scholar"]


class TestCaseMatch:
    """Tests for CaseMatch dataclass."""

    def test_to_dict(self) -> None:
        case = Case(case_id="r1", topic="test")
        match = CaseMatch(case=case, similarity=0.75, match_reasons=["shared terms"])
        d = match.to_dict()
        assert d["similarity"] == 0.75
        assert len(d["match_reasons"]) == 1


class TestStrategyRecommendation:
    """Tests for StrategyRecommendation dataclass."""

    def test_to_dict(self) -> None:
        rec = StrategyRecommendation(
            recommended_sources=["arxiv"],
            confidence=0.8,
            reasoning="Based on 3 cases.",
        )
        d = rec.to_dict()
        assert d["confidence"] == 0.8
        assert "arxiv" in d["recommended_sources"]


# ---------------------------------------------------------------------------
# CaseStore tests
# ---------------------------------------------------------------------------


class TestCaseStore:
    """Tests for CaseStore SQLite persistence."""

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            case = Case(
                case_id="run-1",
                topic="transformer architectures",
                query_terms=["transformer", "attention"],
                sources_used=["arxiv", "scholar"],
                synthesis_quality=0.85,
                outcome="good",
            )
            store.store_case(case)

            retrieved = store.get_case("run-1")
            assert retrieved is not None
            assert retrieved.topic == "transformer architectures"
            assert retrieved.query_terms == ["transformer", "attention"]
            assert retrieved.synthesis_quality == 0.85
        finally:
            store.close()

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            assert store.get_case("nonexistent") is None
        finally:
            store.close()

    def test_get_all_cases(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            for i in range(3):
                store.store_case(Case(case_id=f"r{i}", topic=f"topic {i}"))
            assert len(store.get_all_cases()) == 3
        finally:
            store.close()

    def test_get_successful_cases(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            store.store_case(Case(case_id="r1", topic="t1", synthesis_quality=0.9))
            store.store_case(Case(case_id="r2", topic="t2", synthesis_quality=0.3))
            store.store_case(Case(case_id="r3", topic="t3", synthesis_quality=0.7))

            successful = store.get_successful_cases(min_quality=0.5)
            assert len(successful) == 2
            assert successful[0].case_id == "r1"  # sorted desc
        finally:
            store.close()

    def test_case_count(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            assert store.case_count() == 0
            store.store_case(Case(case_id="r1", topic="t1"))
            assert store.case_count() == 1
        finally:
            store.close()

    def test_store_and_retrieve_adaptation(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            store.store_case(Case(case_id="r1", topic="t1"))
            store.store_case(Case(case_id="r2", topic="t2"))

            adaptation = Adaptation(
                source_case_id="r1",
                target_case_id="r2",
                adaptation_type="source_swap",
                changes_applied={"added_source": "scholar"},
                quality_delta=0.15,
            )
            aid = store.store_adaptation(adaptation)
            assert aid >= 1

            adaptations = store.get_adaptations("r1")
            assert len(adaptations) == 1
            assert adaptations[0].adaptation_type == "source_swap"
            assert adaptations[0].changes_applied["added_source"] == "scholar"
        finally:
            store.close()

    def test_upsert_case(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            store.store_case(Case(case_id="r1", topic="t1", outcome="poor"))
            store.store_case(Case(case_id="r1", topic="t1", outcome="good"))

            case = store.get_case("r1")
            assert case is not None
            assert case.outcome == "good"
            assert store.case_count() == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------


class TestRetrieveSimilarCases:
    """Tests for retrieve_similar_cases."""

    def _make_store(self, tmp_path: Path) -> CaseStore:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        store.store_case(
            Case(
                case_id="r1",
                topic="transformer architectures for time series",
                query_terms=["transformer", "time series", "forecasting"],
                sources_used=["arxiv"],
                synthesis_quality=0.85,
                outcome="good",
            )
        )
        store.store_case(
            Case(
                case_id="r2",
                topic="reinforcement learning in robotics",
                query_terms=["rl", "robot", "control"],
                sources_used=["arxiv", "scholar"],
                synthesis_quality=0.7,
                outcome="adequate",
            )
        )
        store.store_case(
            Case(
                case_id="r3",
                topic="attention mechanisms in neural networks",
                query_terms=["attention", "neural", "transformer"],
                sources_used=["arxiv", "semantic_scholar"],
                synthesis_quality=0.9,
                outcome="excellent",
            )
        )
        return store

    def test_finds_similar_cases(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        try:
            matches = retrieve_similar_cases("transformer models", store)
            assert len(matches) > 0
            # Should match r1 and r3 (both have transformer-related terms)
            case_ids = [m.case.case_id for m in matches]
            assert "r1" in case_ids or "r3" in case_ids
        finally:
            store.close()

    def test_empty_store(self, tmp_path: Path) -> None:
        store = CaseStore(tmp_path / "empty.db")
        try:
            matches = retrieve_similar_cases("anything", store)
            assert matches == []
        finally:
            store.close()

    def test_max_results(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        try:
            matches = retrieve_similar_cases(
                "transformer attention neural", store, max_results=1
            )
            assert len(matches) <= 1
        finally:
            store.close()

    def test_min_quality_filter(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        try:
            matches = retrieve_similar_cases(
                "transformer attention", store, min_quality=0.8
            )
            for match in matches:
                assert match.case.synthesis_quality >= 0.8
        finally:
            store.close()

    def test_similarity_ordering(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        try:
            matches = retrieve_similar_cases(
                "transformer time series forecasting", store
            )
            if len(matches) >= 2:
                assert matches[0].similarity >= matches[1].similarity
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Recommendation tests
# ---------------------------------------------------------------------------


class TestRecommendStrategy:
    """Tests for recommend_strategy."""

    def test_no_matches(self) -> None:
        rec = recommend_strategy("new topic", [])
        assert rec.confidence == 0.0
        assert "defaults" in rec.reasoning.lower()
        assert "arxiv" in rec.recommended_sources

    def test_single_match(self) -> None:
        case = Case(
            case_id="r1",
            topic="transformers",
            sources_used=["arxiv", "scholar"],
            query_terms=["attention", "self-attention"],
            pipeline_profile="deep",
            outcome="excellent",
            synthesis_quality=0.9,
        )
        match = CaseMatch(case=case, similarity=0.8)
        rec = recommend_strategy("transformer models", [match])

        assert rec.confidence > 0
        assert "arxiv" in rec.recommended_sources
        assert rec.recommended_profile == "deep"
        assert len(rec.basis_cases) == 1

    def test_multiple_matches_weighted(self) -> None:
        cases = [
            CaseMatch(
                case=Case(
                    case_id="r1",
                    topic="t1",
                    sources_used=["arxiv"],
                    pipeline_profile="standard",
                    outcome="good",
                    synthesis_quality=0.8,
                ),
                similarity=0.9,
            ),
            CaseMatch(
                case=Case(
                    case_id="r2",
                    topic="t2",
                    sources_used=["arxiv", "scholar", "semantic_scholar"],
                    pipeline_profile="deep",
                    outcome="excellent",
                    synthesis_quality=0.95,
                ),
                similarity=0.7,
            ),
        ]
        rec = recommend_strategy("test", cases)
        assert len(rec.basis_cases) == 2
        assert rec.confidence > 0


# ---------------------------------------------------------------------------
# Case creation from run tests
# ---------------------------------------------------------------------------


class TestCreateCaseFromRun:
    """Tests for create_case_from_run."""

    def test_empty_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "runs" / "run-1").mkdir(parents=True)

        case = create_case_from_run("run-1", "test topic", workspace)
        assert case.case_id == "run-1"
        assert case.topic == "test topic"
        assert case.paper_count == 0

    def test_with_plan(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        plan_dir = workspace / "runs" / "run-1" / "plan"
        plan_dir.mkdir(parents=True)

        plan = {"query_variants": ["transformer", "attention mechanism"]}
        (plan_dir / "query_plan.json").write_text(json.dumps(plan))

        case = create_case_from_run("run-1", "test", workspace)
        assert "transformer" in case.query_terms
        assert "attention mechanism" in case.query_terms

    def test_with_search_results(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        search_dir = workspace / "runs" / "run-1" / "search"
        search_dir.mkdir(parents=True)

        lines = [
            json.dumps({"paper_id": "1", "source": "arxiv"}),
            json.dumps({"paper_id": "2", "source": "scholar"}),
            json.dumps({"paper_id": "3", "source": "arxiv"}),
        ]
        (search_dir / "candidates.jsonl").write_text("\n".join(lines))

        case = create_case_from_run("run-1", "test", workspace)
        assert "arxiv" in case.sources_used
        assert "scholar" in case.sources_used

    def test_with_screen_results(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        screen_dir = workspace / "runs" / "run-1" / "screen"
        screen_dir.mkdir(parents=True)

        lines = [
            json.dumps({"paper_id": "1", "decision": "INCLUDE"}),
            json.dumps({"paper_id": "2", "decision": "EXCLUDE"}),
            json.dumps({"paper_id": "3", "decision": "INCLUDE"}),
        ]
        (screen_dir / "screened.jsonl").write_text("\n".join(lines))

        case = create_case_from_run("run-1", "test", workspace)
        assert case.paper_count == 3
        assert case.shortlist_count == 2

    def test_with_synthesis(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        summarize_dir = workspace / "runs" / "run-1" / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Paper A",
                    "findings": ["F1", "F2", "F3", "F4", "F5"],
                    "evidence": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                }
            ],
            "gaps": [{"description": "Gap 1"}, {"description": "Gap 2"}],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        case = create_case_from_run("run-1", "test", workspace)
        assert case.synthesis_quality > 0


# ---------------------------------------------------------------------------
# Adaptation recording tests
# ---------------------------------------------------------------------------


class TestRecordAdaptation:
    """Tests for record_adaptation."""

    def test_basic_adaptation(self, tmp_path: Path) -> None:
        db = tmp_path / "cbr.db"
        store = CaseStore(db)
        try:
            store.store_case(Case(case_id="r1", topic="t1"))
            store.store_case(Case(case_id="r2", topic="t2"))

            adaptation = record_adaptation(
                store,
                "r1",
                "r2",
                adaptation_type="query_expansion",
                changes={"added_terms": ["new term"]},
                quality_delta=0.1,
            )
            assert adaptation.adaptation_type == "query_expansion"
            assert adaptation.quality_delta == 0.1

            stored = store.get_adaptations("r1")
            assert len(stored) == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# High-level entry point tests
# ---------------------------------------------------------------------------


class TestCbrLookup:
    """Tests for cbr_lookup convenience function."""

    def test_empty_db(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        rec = cbr_lookup("test topic", workspace)
        assert rec.confidence == 0.0
        assert rec.recommended_sources == ["arxiv"]

    def test_with_stored_cases(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        db = workspace / ".cbr_cases.db"

        store = CaseStore(db)
        try:
            store.store_case(
                Case(
                    case_id="r1",
                    topic="machine learning optimization",
                    sources_used=["arxiv", "scholar"],
                    synthesis_quality=0.85,
                    outcome="good",
                )
            )
        finally:
            store.close()

        rec = cbr_lookup("machine learning training", workspace)
        assert rec.confidence > 0
        assert len(rec.basis_cases) > 0


class TestCbrRetain:
    """Tests for cbr_retain convenience function."""

    def test_retain_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        run_dir = workspace / "runs" / "run-1"
        run_dir.mkdir(parents=True)

        case = cbr_retain(
            "run-1",
            "test topic",
            workspace,
            outcome="good",
            strategy_notes="Used arxiv only",
        )
        assert case.case_id == "run-1"
        assert case.outcome == "good"

        # Verify it was stored
        db = workspace / ".cbr_cases.db"
        store = CaseStore(db)
        try:
            assert store.case_count() == 1
            retrieved = store.get_case("run-1")
            assert retrieved is not None
            assert retrieved.outcome == "good"
        finally:
            store.close()
