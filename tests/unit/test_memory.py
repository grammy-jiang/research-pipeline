"""Tests for three-tier memory architecture."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# WorkingMemory tests
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    """Tests for the bounded per-stage working memory."""

    def test_add_and_len(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        assert len(wm) == 0
        wm.add("k1", "v1", stage="plan")
        assert len(wm) == 1
        wm.add("k2", "v2", stage="plan")
        assert len(wm) == 2

    def test_capacity_evicts_oldest(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=3)
        wm.add("a", 1)
        wm.add("b", 2)
        wm.add("c", 3)
        assert len(wm) == 3
        wm.add("d", 4)
        assert len(wm) == 3
        # Oldest ("a") should be evicted
        assert wm.get("a") is None
        assert wm.get("d") is not None
        assert wm.get("d").value == 4  # type: ignore[union-attr]

    def test_capacity_min_one(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=0)
        assert wm.capacity == 1

    def test_get_returns_most_recent(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.add("k", "old")
        wm.add("k", "new")
        item = wm.get("k")
        assert item is not None
        assert item.value == "new"

    def test_get_missing_returns_none(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        assert wm.get("missing") is None

    def test_get_all_ordered(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.add("a", 1)
        wm.add("b", 2)
        wm.add("c", 3)
        items = wm.get_all()
        assert [i.key for i in items] == ["a", "b", "c"]

    def test_get_by_stage(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.add("a", 1, stage="plan")
        wm.add("b", 2, stage="search")
        wm.add("c", 3, stage="plan")
        plan_items = wm.get_by_stage("plan")
        assert len(plan_items) == 2
        assert all(i.stage == "plan" for i in plan_items)

    def test_reset_clears_and_returns(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.add("a", 1, stage="plan")
        wm.add("b", 2, stage="plan")
        cleared = wm.reset("search")
        assert len(cleared) == 2
        assert len(wm) == 0
        assert wm.current_stage == "search"

    def test_reset_sets_new_stage(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        assert wm.current_stage == ""
        wm.reset("plan")
        assert wm.current_stage == "plan"

    def test_add_uses_current_stage(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.reset("search")
        wm.add("k", "v")
        item = wm.get("k")
        assert item is not None
        assert item.stage == "search"

    def test_summary(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=5)
        wm.add("a", 1, stage="plan")
        wm.add("b", 2, stage="search")
        s = wm.summary()
        assert s["capacity"] == 5
        assert s["size"] == 2
        assert set(s["stages"]) == {"plan", "search"}

    def test_metadata(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=10)
        wm.add("k", "v", metadata={"note": "test"})
        item = wm.get("k")
        assert item is not None
        assert item.metadata == {"note": "test"}


# ---------------------------------------------------------------------------
# EpisodicMemory tests
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    """Tests for SQLite-backed episodic memory."""

    def test_record_and_get_roundtrip(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode, EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            ep = Episode(
                run_id="run-001",
                topic="transformer architectures",
                profile="standard",
                started_at="2024-01-01T00:00:00Z",
                completed_at="2024-01-01T01:00:00Z",
                stages_completed=["plan", "search", "screen"],
                paper_count=42,
                shortlist_count=8,
                synthesis_summary="Found key results.",
                gaps_found=["gap1", "gap2"],
                key_decisions=["decision1"],
                outcome="success",
                metadata={"note": "test"},
            )
            mem.record_episode(ep)
            got = mem.get_episode("run-001")
            assert got is not None
            assert got.run_id == "run-001"
            assert got.topic == "transformer architectures"
            assert got.paper_count == 42
            assert got.shortlist_count == 8
            assert got.stages_completed == ["plan", "search", "screen"]
            assert got.gaps_found == ["gap1", "gap2"]
            assert got.key_decisions == ["decision1"]
            assert got.metadata == {"note": "test"}
        finally:
            mem.close()

    def test_search_by_topic(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode, EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            mem.record_episode(
                Episode(
                    run_id="r1",
                    topic="transformer architectures",
                    started_at="2024-01-01",
                )
            )
            mem.record_episode(
                Episode(
                    run_id="r2",
                    topic="graph neural networks",
                    started_at="2024-01-02",
                )
            )
            mem.record_episode(
                Episode(
                    run_id="r3",
                    topic="transformer efficiency",
                    started_at="2024-01-03",
                )
            )
            results = mem.search_by_topic("transformer")
            assert len(results) == 2
            assert {r.run_id for r in results} == {"r1", "r3"}
        finally:
            mem.close()

    def test_recent_episodes_ordered(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode, EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            mem.record_episode(
                Episode(run_id="r1", topic="t1", started_at="2024-01-01")
            )
            mem.record_episode(
                Episode(run_id="r2", topic="t2", started_at="2024-01-03")
            )
            mem.record_episode(
                Episode(run_id="r3", topic="t3", started_at="2024-01-02")
            )
            recent = mem.recent_episodes(limit=2)
            assert len(recent) == 2
            assert recent[0].run_id == "r2"
            assert recent[1].run_id == "r3"
        finally:
            mem.close()

    def test_count(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode, EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            assert mem.count() == 0
            mem.record_episode(
                Episode(run_id="r1", topic="t1", started_at="2024-01-01")
            )
            assert mem.count() == 1
            mem.record_episode(
                Episode(run_id="r2", topic="t2", started_at="2024-01-02")
            )
            assert mem.count() == 2
        finally:
            mem.close()

    def test_update_existing_episode(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode, EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            mem.record_episode(
                Episode(
                    run_id="r1",
                    topic="transformers",
                    started_at="2024-01-01",
                    paper_count=10,
                )
            )
            mem.record_episode(
                Episode(
                    run_id="r1",
                    topic="transformers",
                    started_at="2024-01-01",
                    paper_count=42,
                    outcome="updated",
                )
            )
            assert mem.count() == 1
            got = mem.get_episode("r1")
            assert got is not None
            assert got.paper_count == 42
            assert got.outcome == "updated"
        finally:
            mem.close()

    def test_empty_search_returns_empty(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            results = mem.search_by_topic("nonexistent")
            assert results == []
        finally:
            mem.close()

    def test_get_missing_episode_returns_none(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import EpisodicMemory

        db = tmp_path / "ep.db"
        mem = EpisodicMemory(db_path=db)
        try:
            assert mem.get_episode("missing") is None
        finally:
            mem.close()


# ---------------------------------------------------------------------------
# SemanticMemory tests
# ---------------------------------------------------------------------------


class TestSemanticMemory:
    """Tests for KG-backed semantic memory."""

    def test_known_concepts_empty(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory

        kg_db = tmp_path / "kg.db"
        sm = SemanticMemory(kg_path=kg_db)
        try:
            concepts = sm.known_concepts()
            assert concepts == []
        finally:
            sm.close()

    def test_known_methods_empty(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory

        kg_db = tmp_path / "kg.db"
        sm = SemanticMemory(kg_path=kg_db)
        try:
            methods = sm.known_methods()
            assert methods == []
        finally:
            sm.close()

    def test_topic_overlap_returns_dict(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory

        kg_db = tmp_path / "kg.db"
        sm = SemanticMemory(kg_path=kg_db)
        try:
            overlap = sm.topic_overlap("transformer attention")
            assert isinstance(overlap, dict)
            assert "concepts" in overlap
            assert "methods" in overlap
            assert "papers" in overlap
        finally:
            sm.close()

    def test_stats_returns_dict(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory

        kg_db = tmp_path / "kg.db"
        sm = SemanticMemory(kg_path=kg_db)
        try:
            stats = sm.stats()
            assert isinstance(stats, dict)
            assert "total_entities" in stats
        finally:
            sm.close()

    def test_related_papers_no_match(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory

        kg_db = tmp_path / "kg.db"
        sm = SemanticMemory(kg_path=kg_db)
        try:
            papers = sm.related_papers("nonexistent_concept")
            assert papers == []
        finally:
            sm.close()

    def test_known_concepts_with_data(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory
        from research_pipeline.storage.knowledge_graph import (
            EntityType,
            KnowledgeGraph,
        )

        kg_db = tmp_path / "kg.db"
        kg = KnowledgeGraph(db_path=kg_db)
        kg.add_entity("concept:attention", EntityType.CONCEPT, "attention mechanism")
        kg.add_entity("concept:transformer", EntityType.CONCEPT, "transformer")
        kg.close()

        sm = SemanticMemory(kg_path=kg_db)
        try:
            concepts = sm.known_concepts()
            assert len(concepts) == 2
            names = {c["name"] for c in concepts}
            assert "attention mechanism" in names
            assert "transformer" in names
        finally:
            sm.close()

    def test_known_concepts_limit(self, tmp_path: Path) -> None:
        from research_pipeline.memory.semantic import SemanticMemory
        from research_pipeline.storage.knowledge_graph import (
            EntityType,
            KnowledgeGraph,
        )

        kg_db = tmp_path / "kg.db"
        kg = KnowledgeGraph(db_path=kg_db)
        for i in range(10):
            kg.add_entity(f"concept:{i}", EntityType.CONCEPT, f"concept_{i}")
        kg.close()

        sm = SemanticMemory(kg_path=kg_db)
        try:
            concepts = sm.known_concepts(limit=3)
            assert len(concepts) == 3
        finally:
            sm.close()


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------


class TestMemoryManager:
    """Tests for the coordinating memory manager."""

    def test_construction(self, tmp_path: Path) -> None:
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(
            working_capacity=25,
            episodic_path=ep_db,
            kg_path=kg_db,
        )
        try:
            assert mgr.working.capacity == 25
        finally:
            mgr.close()

    def test_transition_stage_resets_working(self, tmp_path: Path) -> None:
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(episodic_path=ep_db, kg_path=kg_db)
        try:
            mgr.working.add("a", 1, stage="plan")
            mgr.working.add("b", 2, stage="plan")
            cleared = mgr.transition_stage("search")
            assert len(cleared) == 2
            assert len(mgr.working) == 0
            assert mgr.working.current_stage == "search"
        finally:
            mgr.close()

    def test_record_run_stores_episode(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(episodic_path=ep_db, kg_path=kg_db)
        try:
            ep = Episode(
                run_id="r1",
                topic="test topic",
                started_at="2024-01-01",
            )
            mgr.record_run(ep)
            assert mgr.episodic.count() == 1
            got = mgr.episodic.get_episode("r1")
            assert got is not None
            assert got.topic == "test topic"
        finally:
            mgr.close()

    def test_prior_knowledge(self, tmp_path: Path) -> None:
        from research_pipeline.memory.episodic import Episode
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(episodic_path=ep_db, kg_path=kg_db)
        try:
            mgr.episodic.record_episode(
                Episode(
                    run_id="r1",
                    topic="transformer attention",
                    started_at="2024-01-01",
                )
            )
            prior = mgr.prior_knowledge("transformer")
            assert prior["past_runs"] == 1
            assert "r1" in prior["past_run_ids"]
            assert isinstance(prior["known_concepts"], int)
        finally:
            mgr.close()

    def test_summary_returns_all_tiers(self, tmp_path: Path) -> None:
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(episodic_path=ep_db, kg_path=kg_db)
        try:
            s = mgr.summary()
            assert "working" in s
            assert "episodic" in s
            assert "semantic" in s
            assert s["episodic"]["total_episodes"] == 0
        finally:
            mgr.close()

    def test_close_is_safe(self, tmp_path: Path) -> None:
        from research_pipeline.memory.manager import MemoryManager

        kg_db = tmp_path / "kg.db"
        ep_db = tmp_path / "ep.db"
        mgr = MemoryManager(episodic_path=ep_db, kg_path=kg_db)
        mgr.close()
        # Double close should not raise
        mgr.close()
