"""Tests for memory consolidation engine (B4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from research_pipeline.pipeline.consolidation import (
    ConsolidationResult,
    DriftMetric,
    Episode,
    EpisodeStore,
    Rule,
    consolidate,
    extract_episode_from_run,
    find_recurring_findings,
    measure_drift,
    promote_to_rules,
    prune_stale_episodes,
    run_consolidation,
)

# ── Episode dataclass tests ────────────────────────────────────────


class TestEpisode:
    def test_basic_creation(self) -> None:
        ep = Episode(
            run_id="r1",
            topic="transformers",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        assert ep.run_id == "r1"
        assert ep.topic == "transformers"
        assert ep.findings == []
        assert ep.agreements == []
        assert ep.paper_count == 0

    def test_content_hash_deterministic(self) -> None:
        ep = Episode(
            run_id="r1",
            topic="topic",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["f1", "f2"],
            agreements=["a1"],
        )
        h1 = ep.content_hash()
        h2 = ep.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_differs_for_different_content(self) -> None:
        ep1 = Episode(
            run_id="r1",
            topic="topic",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["f1"],
        )
        ep2 = Episode(
            run_id="r1",
            topic="topic",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["f2"],
        )
        assert ep1.content_hash() != ep2.content_hash()

    def test_content_hash_order_independent(self) -> None:
        ep1 = Episode(
            run_id="r1",
            topic="t",
            timestamp="t",
            findings=["a", "b"],
        )
        ep2 = Episode(
            run_id="r1",
            topic="t",
            timestamp="t",
            findings=["b", "a"],
        )
        assert ep1.content_hash() == ep2.content_hash()


# ── Rule dataclass tests ───────────────────────────────────────────


class TestRule:
    def test_basic_creation(self) -> None:
        r = Rule(rule_id="rule-1", statement="finding text")
        assert r.rule_id == "rule-1"
        assert r.statement == "finding text"
        assert r.confidence == 0.0
        assert r.supporting_runs == []

    def test_with_full_fields(self) -> None:
        r = Rule(
            rule_id="rule-2",
            statement="stmt",
            supporting_runs=["r1", "r2"],
            confidence=0.8,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-06-01T00:00:00+00:00",
            source_findings=["f1"],
        )
        assert r.confidence == 0.8
        assert len(r.supporting_runs) == 2


# ── DriftMetric tests ──────────────────────────────────────────────


class TestDriftMetric:
    def test_basic(self) -> None:
        d = DriftMetric(run_a="r1", run_b="r2")
        assert d.finding_overlap == 0.0
        assert d.topic_shift == 0.0
        assert d.drift_score == 0.0


# ── EpisodeStore tests ─────────────────────────────────────────────


class TestEpisodeStore:
    def test_create_and_get_episode(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        ep = Episode(
            run_id="r1",
            topic="AI memory",
            timestamp="2025-01-01T00:00:00+00:00",
            paper_count=5,
            findings=["f1", "f2"],
            agreements=["a1"],
            open_questions=["q1"],
        )
        store.add_episode(ep)
        got = store.get_episode("r1")
        assert got is not None
        assert got.run_id == "r1"
        assert got.topic == "AI memory"
        assert got.paper_count == 5
        assert got.findings == ["f1", "f2"]
        assert got.agreements == ["a1"]
        assert got.open_questions == ["q1"]
        store.close()

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        assert store.get_episode("missing") is None
        store.close()

    def test_add_episode_upsert(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        ep1 = Episode(
            run_id="r1",
            topic="old",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        store.add_episode(ep1)
        ep2 = Episode(
            run_id="r1",
            topic="new",
            timestamp="2025-02-01T00:00:00+00:00",
        )
        store.add_episode(ep2)
        got = store.get_episode("r1")
        assert got is not None
        assert got.topic == "new"
        store.close()

    def test_list_episodes(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        for i in range(3):
            store.add_episode(
                Episode(
                    run_id=f"r{i}",
                    topic="t",
                    timestamp=f"2025-0{i + 1}-01T00:00:00+00:00",
                )
            )
        eps = store.list_episodes()
        assert len(eps) == 3
        assert eps[0].run_id == "r0"
        store.close()

    def test_list_episodes_with_limit(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        for i in range(5):
            store.add_episode(
                Episode(
                    run_id=f"r{i}",
                    topic="t",
                    timestamp=f"2025-0{i + 1}-01T00:00:00+00:00",
                )
            )
        eps = store.list_episodes(limit=2)
        assert len(eps) == 2
        store.close()

    def test_list_episodes_consolidated_filter(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        for i in range(3):
            store.add_episode(
                Episode(
                    run_id=f"r{i}",
                    topic="t",
                    timestamp=f"2025-0{i + 1}-01T00:00:00+00:00",
                )
            )
        store.mark_consolidated(["r0"])
        cons = store.list_episodes(consolidated=True)
        assert len(cons) == 1
        assert cons[0].run_id == "r0"
        uncons = store.list_episodes(consolidated=False)
        assert len(uncons) == 2
        store.close()

    def test_count_episodes(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        assert store.count_episodes() == 0
        store.add_episode(
            Episode(run_id="r1", topic="t", timestamp="2025-01-01T00:00:00+00:00")
        )
        assert store.count_episodes() == 1
        assert store.count_episodes(consolidated=False) == 1
        assert store.count_episodes(consolidated=True) == 0
        store.close()

    def test_mark_consolidated(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        store.add_episode(
            Episode(run_id="r1", topic="t", timestamp="2025-01-01T00:00:00+00:00")
        )
        store.add_episode(
            Episode(run_id="r2", topic="t", timestamp="2025-02-01T00:00:00+00:00")
        )
        store.mark_consolidated(["r1"])
        assert store.count_episodes(consolidated=True) == 1
        assert store.count_episodes(consolidated=False) == 1
        store.close()

    def test_mark_consolidated_empty(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        store.mark_consolidated([])  # should not raise
        store.close()

    def test_delete_episodes(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        for i in range(3):
            store.add_episode(
                Episode(
                    run_id=f"r{i}",
                    topic="t",
                    timestamp=f"2025-0{i + 1}-01T00:00:00+00:00",
                )
            )
        deleted = store.delete_episodes(["r0", "r1"])
        assert deleted == 2
        assert store.count_episodes() == 1
        store.close()

    def test_delete_episodes_empty(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        assert store.delete_episodes([]) == 0
        store.close()

    def test_add_and_get_rule(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        r = Rule(
            rule_id="rule-1",
            statement="transformers are effective",
            supporting_runs=["r1", "r2"],
            confidence=0.6,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
            source_findings=["sf1"],
        )
        store.add_rule(r)
        got = store.get_rule("rule-1")
        assert got is not None
        assert got.statement == "transformers are effective"
        assert got.confidence == 0.6
        assert got.supporting_runs == ["r1", "r2"]
        store.close()

    def test_get_rule_nonexistent(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        assert store.get_rule("missing") is None
        store.close()

    def test_list_rules(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        for i, conf in enumerate([0.3, 0.7, 0.5]):
            store.add_rule(
                Rule(
                    rule_id=f"r{i}",
                    statement=f"stmt{i}",
                    confidence=conf,
                    created_at="2025-01-01T00:00:00+00:00",
                    updated_at="2025-01-01T00:00:00+00:00",
                )
            )
        all_rules = store.list_rules()
        assert len(all_rules) == 3
        # sorted by confidence descending
        assert all_rules[0].confidence == 0.7

        filtered = store.list_rules(min_confidence=0.5)
        assert len(filtered) == 2
        store.close()

    def test_delete_rules(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        store.add_rule(
            Rule(
                rule_id="r1",
                statement="s",
                created_at="t",
                updated_at="t",
            )
        )
        deleted = store.delete_rules(["r1"])
        assert deleted == 1
        assert store.get_rule("r1") is None
        store.close()

    def test_delete_rules_empty(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        assert store.delete_rules([]) == 0
        store.close()

    def test_log_and_list_drift(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        d = DriftMetric(
            run_a="r1",
            run_b="r2",
            finding_overlap=0.5,
            topic_shift=0.1,
            drift_score=0.3,
        )
        store.log_drift(d)
        drifts = store.list_drift()
        assert len(drifts) == 1
        assert drifts[0].run_a == "r1"
        assert drifts[0].finding_overlap == 0.5
        store.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        store.close()
        store.close()  # should not raise


# ── extract_episode_from_run tests ─────────────────────────────────


def _write_synthesis(run_dir: Path, data: dict) -> None:
    """Helper to write a synthesis report."""
    synth_dir = run_dir / "summarize"
    synth_dir.mkdir(parents=True, exist_ok=True)
    (synth_dir / "synthesis_report.json").write_text(json.dumps(data))


class TestExtractEpisodeFromRun:
    def test_extract_from_synthesis(self, tmp_path: Path) -> None:
        run = tmp_path / "run1"
        _write_synthesis(
            run,
            {
                "topic": "memory systems",
                "paper_count": 10,
                "agreements": [
                    {"claim": "memory improves performance"},
                    {"claim": "consolidation is needed"},
                ],
                "disagreements": [{"topic": "optimal architecture"}],
                "open_questions": ["what about scale?"],
            },
        )
        ep = extract_episode_from_run(tmp_path, "run1")
        assert ep is not None
        assert ep.run_id == "run1"
        assert ep.topic == "memory systems"
        assert ep.paper_count == 10
        assert len(ep.findings) == 3  # 2 agreements + 1 disagreement
        assert len(ep.agreements) == 2
        assert len(ep.open_questions) == 1

    def test_missing_run_dir(self, tmp_path: Path) -> None:
        ep = extract_episode_from_run(tmp_path, "missing")
        assert ep is None

    def test_no_synthesis_file(self, tmp_path: Path) -> None:
        (tmp_path / "run1").mkdir()
        ep = extract_episode_from_run(tmp_path, "run1")
        assert ep is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        run = tmp_path / "run1"
        synth_dir = run / "summarize"
        synth_dir.mkdir(parents=True)
        (synth_dir / "synthesis_report.json").write_text("not json")
        ep = extract_episode_from_run(tmp_path, "run1")
        assert ep is None

    def test_empty_synthesis(self, tmp_path: Path) -> None:
        run = tmp_path / "run1"
        _write_synthesis(run, {})
        ep = extract_episode_from_run(tmp_path, "run1")
        assert ep is not None
        assert ep.topic == ""
        assert ep.paper_count == 0
        assert ep.findings == []


# ── measure_drift tests ────────────────────────────────────────────


class TestMeasureDrift:
    def test_identical_episodes(self) -> None:
        ep = Episode(
            run_id="r1",
            topic="transformers",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["transformers work well"],
        )
        d = measure_drift(ep, ep)
        assert d.finding_overlap == 1.0
        assert d.topic_shift == 0.0
        assert d.drift_score == 0.0

    def test_completely_different(self) -> None:
        ep_a = Episode(
            run_id="r1",
            topic="quantum computing",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["qubits enable parallelism"],
        )
        ep_b = Episode(
            run_id="r2",
            topic="marine biology",
            timestamp="2025-02-01T00:00:00+00:00",
            findings=["coral reefs declining"],
        )
        d = measure_drift(ep_a, ep_b)
        assert d.finding_overlap < 0.2
        assert d.topic_shift > 0.5
        assert d.drift_score > 0.5

    def test_empty_findings(self) -> None:
        ep_a = Episode(
            run_id="r1",
            topic="t",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        ep_b = Episode(
            run_id="r2",
            topic="t",
            timestamp="2025-02-01T00:00:00+00:00",
        )
        d = measure_drift(ep_a, ep_b)
        # Empty findings → overlap is 1.0 (both empty)
        assert d.finding_overlap == 1.0
        assert d.drift_score == 0.0

    def test_partial_overlap(self) -> None:
        ep_a = Episode(
            run_id="r1",
            topic="neural networks",
            timestamp="2025-01-01T00:00:00+00:00",
            findings=["attention mechanism effective", "dropout prevents overfitting"],
        )
        ep_b = Episode(
            run_id="r2",
            topic="neural networks",
            timestamp="2025-02-01T00:00:00+00:00",
            findings=[
                "attention mechanism effective",
                "batch normalization helps training",
            ],
        )
        d = measure_drift(ep_a, ep_b)
        assert 0.0 < d.finding_overlap < 1.0
        assert d.topic_shift == 0.0  # same topic


# ── find_recurring_findings tests ──────────────────────────────────


class TestFindRecurringFindings:
    def test_basic_recurrence(self) -> None:
        episodes = [
            Episode(
                run_id="r1",
                topic="t",
                timestamp="t",
                findings=["transformers outperform baselines"],
            ),
            Episode(
                run_id="r2",
                topic="t",
                timestamp="t",
                findings=["transformers outperform baselines"],
            ),
        ]
        recurring = find_recurring_findings(episodes, min_support=2)
        assert len(recurring) >= 1
        _, runs = recurring[0]
        assert "r1" in runs
        assert "r2" in runs

    def test_no_recurrence(self) -> None:
        episodes = [
            Episode(
                run_id="r1",
                topic="t",
                timestamp="t",
                findings=["finding A unique"],
            ),
            Episode(
                run_id="r2",
                topic="t",
                timestamp="t",
                findings=["finding B different"],
            ),
        ]
        recurring = find_recurring_findings(episodes, min_support=2)
        assert len(recurring) == 0

    def test_fuzzy_matching(self) -> None:
        episodes = [
            Episode(
                run_id="r1",
                topic="t",
                timestamp="t",
                findings=["transformer models achieve good performance"],
            ),
            Episode(
                run_id="r2",
                topic="t",
                timestamp="t",
                findings=["transformer models achieve excellent performance"],
            ),
        ]
        recurring = find_recurring_findings(episodes, min_support=2)
        # High Jaccard overlap should match
        assert len(recurring) >= 1

    def test_dedup_within_episode(self) -> None:
        episodes = [
            Episode(
                run_id="r1",
                topic="t",
                timestamp="t",
                findings=["same finding", "same finding"],
            ),
            Episode(
                run_id="r2",
                topic="t",
                timestamp="t",
                findings=["same finding"],
            ),
        ]
        recurring = find_recurring_findings(episodes, min_support=2)
        assert len(recurring) >= 1
        _, runs = recurring[0]
        # r1 should only appear once
        assert runs.count("r1") == 1

    def test_min_support_threshold(self) -> None:
        episodes = [
            Episode(
                run_id=f"r{i}",
                topic="t",
                timestamp="t",
                findings=["common finding everywhere"],
            )
            for i in range(5)
        ]
        rec2 = find_recurring_findings(episodes, min_support=2)
        rec4 = find_recurring_findings(episodes, min_support=4)
        assert len(rec2) >= 1
        assert len(rec4) >= 1
        # Higher threshold still met
        _, runs = rec4[0]
        assert len(runs) >= 4

    def test_empty_episodes(self) -> None:
        assert find_recurring_findings([], min_support=2) == []

    def test_sorted_by_support_count(self) -> None:
        episodes = [
            Episode(
                run_id="r1",
                topic="t",
                timestamp="t",
                findings=["common finding", "rare finding alpha"],
            ),
            Episode(
                run_id="r2",
                topic="t",
                timestamp="t",
                findings=["common finding", "rare finding alpha"],
            ),
            Episode(
                run_id="r3",
                topic="t",
                timestamp="t",
                findings=["common finding"],
            ),
        ]
        recurring = find_recurring_findings(episodes, min_support=2)
        if len(recurring) >= 2:
            # First result should have more support
            assert len(recurring[0][1]) >= len(recurring[1][1])


# ── promote_to_rules tests ─────────────────────────────────────────


class TestPromoteToRules:
    def test_create_new_rule(self) -> None:
        recurring = [("attention is effective", ["r1", "r2", "r3"])]
        new, updated = promote_to_rules(recurring, [])
        assert len(new) == 1
        assert len(updated) == 0
        assert new[0].statement == "attention is effective"
        assert new[0].confidence == 0.6  # 3/5
        assert "r1" in new[0].supporting_runs

    def test_update_existing_rule(self) -> None:
        existing = Rule(
            rule_id="rule-old",
            statement="attention is effective",
            supporting_runs=["r1"],
            confidence=0.2,
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        recurring = [("attention mechanism effective", ["r2", "r3"])]
        new, updated = promote_to_rules(recurring, [existing])
        assert len(new) == 0
        assert len(updated) == 1
        assert "r2" in updated[0].supporting_runs
        assert updated[0].confidence > 0.2

    def test_confidence_capped_at_1(self) -> None:
        recurring = [("finding", [f"r{i}" for i in range(10)])]
        new, _ = promote_to_rules(recurring, [])
        assert new[0].confidence == 1.0

    def test_empty_recurring(self) -> None:
        new, updated = promote_to_rules([], [])
        assert new == []
        assert updated == []


# ── prune_stale_episodes tests ─────────────────────────────────────


class TestPruneStaleEpisodes:
    def test_prune_old_covered_episode(self) -> None:
        old_ts = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        episodes = [
            Episode(
                run_id="old-run",
                topic="t",
                timestamp=old_ts,
                findings=["transformers work well"],
            )
        ]
        rules = [
            Rule(
                rule_id="r1",
                statement="transformers work well",
                created_at="t",
                updated_at="t",
            )
        ]
        prunable = prune_stale_episodes(episodes, rules, staleness_days=90)
        assert "old-run" in prunable

    def test_keep_recent_episode(self) -> None:
        recent_ts = datetime.now(UTC).isoformat()
        episodes = [
            Episode(
                run_id="new-run",
                topic="t",
                timestamp=recent_ts,
                findings=["transformers work well"],
            )
        ]
        rules = [
            Rule(
                rule_id="r1",
                statement="transformers work well",
                created_at="t",
                updated_at="t",
            )
        ]
        prunable = prune_stale_episodes(episodes, rules, staleness_days=90)
        assert "new-run" not in prunable

    def test_keep_old_uncovered_episode(self) -> None:
        old_ts = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        episodes = [
            Episode(
                run_id="old-unique",
                topic="t",
                timestamp=old_ts,
                findings=["unique finding not in any rule"],
            )
        ]
        rules = [
            Rule(
                rule_id="r1",
                statement="completely different statement",
                created_at="t",
                updated_at="t",
            )
        ]
        prunable = prune_stale_episodes(episodes, rules, staleness_days=90)
        assert "old-unique" not in prunable

    def test_prune_old_empty_findings(self) -> None:
        old_ts = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        episodes = [
            Episode(
                run_id="empty-old",
                topic="t",
                timestamp=old_ts,
                findings=[],
            )
        ]
        prunable = prune_stale_episodes(episodes, [], staleness_days=90)
        assert "empty-old" in prunable

    def test_invalid_timestamp_skipped(self) -> None:
        episodes = [
            Episode(
                run_id="bad-ts",
                topic="t",
                timestamp="not-a-date",
                findings=["f"],
            )
        ]
        prunable = prune_stale_episodes(episodes, [], staleness_days=90)
        assert prunable == []


# ── consolidate (orchestrator) tests ───────────────────────────────


class TestConsolidate:
    def _make_store_with_episodes(self, tmp_path: Path, count: int) -> EpisodeStore:
        store = EpisodeStore(tmp_path / "test.db")
        for i in range(count):
            store.add_episode(
                Episode(
                    run_id=f"r{i}",
                    topic="neural networks",
                    timestamp=f"2025-{(i % 12) + 1:02d}-01T00:00:00+00:00",
                    findings=[
                        "transformers effective",
                        f"finding unique to r{i}",
                    ],
                )
            )
        return store

    def test_below_threshold_skips(self, tmp_path: Path) -> None:
        store = self._make_store_with_episodes(tmp_path, 5)
        result = consolidate(store, capacity=100, threshold=0.8)
        assert result.episodes_before == 5
        assert result.episodes_after == 5
        assert result.rules_created == 0
        store.close()

    def test_above_threshold_consolidates(self, tmp_path: Path) -> None:
        store = self._make_store_with_episodes(tmp_path, 10)
        result = consolidate(store, capacity=10, threshold=0.8, min_support=2)
        assert result.episodes_before == 10
        assert result.rules_created > 0
        store.close()

    def test_dry_run_no_modification(self, tmp_path: Path) -> None:
        store = self._make_store_with_episodes(tmp_path, 10)
        result = consolidate(
            store,
            capacity=10,
            threshold=0.8,
            min_support=2,
            dry_run=True,
        )
        assert result.rules_created > 0
        # But store should be unchanged
        assert store.count_episodes(consolidated=True) == 0
        assert len(store.list_rules()) == 0
        store.close()

    def test_drift_metrics_computed(self, tmp_path: Path) -> None:
        store = self._make_store_with_episodes(tmp_path, 5)
        result = consolidate(store, capacity=100, threshold=0.8)
        # Should have N-1 drift measurements
        assert len(result.drift_metrics) == 4
        store.close()

    def test_empty_store(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        result = consolidate(store)
        assert result.episodes_before == 0
        assert result.episodes_after == 0
        store.close()

    def test_timestamp_in_result(self, tmp_path: Path) -> None:
        store = EpisodeStore(tmp_path / "test.db")
        result = consolidate(store)
        assert result.timestamp != ""
        store.close()


# ── run_consolidation (high-level) tests ───────────────────────────


class TestRunConsolidation:
    def _setup_runs(self, workspace: Path, count: int) -> list[str]:
        run_ids = []
        for i in range(count):
            rid = f"run{i}"
            run_ids.append(rid)
            _write_synthesis(
                workspace / rid,
                {
                    "topic": "neural networks",
                    "paper_count": 5,
                    "agreements": [
                        {"claim": "attention mechanism effective"},
                        {"claim": f"unique finding {i}"},
                    ],
                    "disagreements": [],
                    "open_questions": [],
                },
            )
        return run_ids

    def test_basic_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        run_ids = self._setup_runs(workspace, 3)
        result = run_consolidation(workspace, run_ids, db_path=tmp_path / "cons.db")
        assert result.episodes_before == 3

    def test_with_output(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        self._setup_runs(workspace, 3)
        out = tmp_path / "output" / "report.json"
        run_consolidation(
            workspace,
            db_path=tmp_path / "cons.db",
            output=out,
        )
        assert out.exists()
        data = json.loads(out.read_text())
        assert "episodes_before" in data
        assert "timestamp" in data

    def test_auto_scan_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        self._setup_runs(workspace, 2)
        result = run_consolidation(workspace, db_path=tmp_path / "cons.db")
        assert result.episodes_before == 2

    def test_no_runs(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        workspace.mkdir(parents=True)
        result = run_consolidation(workspace, [], db_path=tmp_path / "cons.db")
        assert result.episodes_before == 0

    def test_idempotent_ingestion(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        run_ids = self._setup_runs(workspace, 2)
        db = tmp_path / "cons.db"
        r1 = run_consolidation(workspace, run_ids, db_path=db)
        r2 = run_consolidation(workspace, run_ids, db_path=db)
        assert r1.episodes_before == r2.episodes_before

    def test_missing_workspace(self, tmp_path: Path) -> None:
        result = run_consolidation(
            tmp_path / "nonexistent",
            db_path=tmp_path / "cons.db",
        )
        assert result.episodes_before == 0

    def test_dry_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "ws"
        self._setup_runs(workspace, 3)
        result = run_consolidation(
            workspace,
            db_path=tmp_path / "cons.db",
            dry_run=True,
        )
        assert result.episodes_before == 3


# ── ConsolidationResult tests ──────────────────────────────────────


class TestConsolidationResult:
    def test_defaults(self) -> None:
        r = ConsolidationResult()
        assert r.episodes_before == 0
        assert r.episodes_after == 0
        assert r.rules_created == 0
        assert r.rules_updated == 0
        assert r.entries_pruned == 0
        assert r.drift_metrics == []
        assert r.timestamp == ""
