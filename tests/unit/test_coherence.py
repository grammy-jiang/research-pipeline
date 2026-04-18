"""Tests for multi-session coherence evaluation."""

import json
from pathlib import Path

import pytest

from research_pipeline.pipeline.coherence import (
    CONTRADICTION_SIMILARITY_THRESHOLD,
    TOPIC_SIMILARITY_THRESHOLD,
    CoherenceReport,
    CoherenceScore,
    Contradiction,
    Finding,
    KnowledgeUpdate,
    compute_factual_consistency,
    compute_knowledge_update_fidelity,
    compute_temporal_ordering,
    compute_topic_overlap,
    detect_contradictions,
    evaluate_coherence,
    extract_findings,
    jaccard_similarity,
    load_run_timestamp,
    load_synthesis,
    run_coherence,
    track_knowledge_updates,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_synthesis(
    findings: dict[str, list[str | dict]] | None = None,
    themes: list[dict] | None = None,
    gaps: list[dict] | None = None,
) -> dict:
    """Build a minimal synthesis_results.json dict."""
    result: dict = {}
    if findings:
        result["confidence_graded_findings"] = findings
    if themes:
        result["themes"] = themes
    if gaps:
        result["gaps"] = gaps
    return result


def _write_synthesis(run_root: Path, synthesis: dict) -> None:
    """Write synthesis results to expected location."""
    synth_dir = run_root / "summarize"
    synth_dir.mkdir(parents=True, exist_ok=True)
    (synth_dir / "synthesis_results.json").write_text(json.dumps(synthesis, indent=2))


def _write_manifest(run_root: Path, started_at: str = "2025-01-01T00:00:00") -> None:
    """Write a run manifest with timestamp."""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "run_manifest.json").write_text(json.dumps({"started_at": started_at}))


# ===========================================================================
# Finding dataclass tests
# ===========================================================================


class TestFinding:
    """Tests for the Finding dataclass."""

    def test_creation_defaults(self) -> None:
        f = Finding(text="some finding")
        assert f.text == "some finding"
        assert f.confidence == "medium"
        assert f.run_id == ""
        assert f.timestamp == ""
        assert f.evidence_ids == []

    def test_creation_full(self) -> None:
        f = Finding(
            text="detailed finding",
            confidence="high",
            run_id="run1",
            timestamp="2025-01-01",
            evidence_ids=["e1", "e2"],
        )
        assert f.confidence == "high"
        assert f.evidence_ids == ["e1", "e2"]

    def test_to_dict(self) -> None:
        f = Finding(text="test", confidence="low", run_id="r1")
        d = f.to_dict()
        assert d["text"] == "test"
        assert d["confidence"] == "low"
        assert d["run_id"] == "r1"

    def test_evidence_ids_independent(self) -> None:
        f1 = Finding(text="a")
        f2 = Finding(text="b")
        f1.evidence_ids.append("e1")
        assert f2.evidence_ids == []


# ===========================================================================
# Contradiction dataclass tests
# ===========================================================================


class TestContradiction:
    """Tests for the Contradiction dataclass."""

    def test_to_dict(self) -> None:
        c = Contradiction(
            finding_a=Finding(text="A"),
            finding_b=Finding(text="B"),
            similarity=0.75,
            explanation="test",
        )
        d = c.to_dict()
        assert d["similarity"] == 0.75
        assert d["finding_a"]["text"] == "A"
        assert d["finding_b"]["text"] == "B"


# ===========================================================================
# KnowledgeUpdate dataclass tests
# ===========================================================================


class TestKnowledgeUpdate:
    """Tests for the KnowledgeUpdate dataclass."""

    def test_new_update(self) -> None:
        u = KnowledgeUpdate(topic="test", new_finding=Finding(text="new"))
        d = u.to_dict()
        assert d["update_type"] == "new"
        assert d["old_finding"] is None
        assert d["new_finding"]["text"] == "new"

    def test_revised_update(self) -> None:
        u = KnowledgeUpdate(
            topic="test",
            old_finding=Finding(text="old"),
            new_finding=Finding(text="new"),
            update_type="revised",
        )
        d = u.to_dict()
        assert d["update_type"] == "revised"
        assert d["old_finding"]["text"] == "old"


# ===========================================================================
# CoherenceScore and CoherenceReport tests
# ===========================================================================


class TestCoherenceScore:
    """Tests for CoherenceScore."""

    def test_to_dict(self) -> None:
        s = CoherenceScore(
            factual_consistency=0.9,
            temporal_ordering=0.8,
            knowledge_update_fidelity=0.7,
            contradiction_rate=0.1,
            overall=0.85,
        )
        d = s.to_dict()
        assert d["factual_consistency"] == 0.9
        assert d["overall"] == 0.85


class TestCoherenceReport:
    """Tests for CoherenceReport."""

    def test_to_dict_minimal(self) -> None:
        r = CoherenceReport(
            run_ids=["r1", "r2"],
            score=CoherenceScore(overall=0.9),
        )
        d = r.to_dict()
        assert d["run_ids"] == ["r1", "r2"]
        assert d["score"]["overall"] == 0.9
        assert d["contradictions"] == []
        assert d["knowledge_updates"] == []

    def test_to_dict_with_data(self) -> None:
        r = CoherenceReport(
            run_ids=["r1", "r2"],
            score=CoherenceScore(overall=0.7),
            contradictions=[
                Contradiction(
                    finding_a=Finding(text="A"),
                    finding_b=Finding(text="B"),
                )
            ],
            finding_count=10,
            common_finding_count=3,
            topic_overlap=0.5,
        )
        d = r.to_dict()
        assert len(d["contradictions"]) == 1
        assert d["finding_count"] == 10
        assert d["topic_overlap"] == 0.5


# ===========================================================================
# Jaccard similarity tests
# ===========================================================================


class TestJaccardSimilarity:
    """Tests for jaccard_similarity."""

    def test_identical(self) -> None:
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert jaccard_similarity("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap(self) -> None:
        sim = jaccard_similarity(
            "transformer architecture for time series",
            "transformer model for classification",
        )
        assert 0.0 < sim < 1.0

    def test_empty_strings(self) -> None:
        assert jaccard_similarity("", "") == 0.0

    def test_case_insensitive(self) -> None:
        assert jaccard_similarity("Hello World", "hello world") == 1.0

    def test_stop_words_removed(self) -> None:
        # "the" and "a" are stop words, should be removed
        sim = jaccard_similarity("the cat", "a cat")
        assert sim == 1.0


# ===========================================================================
# extract_findings tests
# ===========================================================================


class TestExtractFindings:
    """Tests for extract_findings."""

    def test_empty_synthesis(self) -> None:
        findings = extract_findings({})
        assert findings == []

    def test_confidence_graded_dict_findings(self) -> None:
        synth = _make_synthesis(
            findings={
                "high": [{"finding": "Important result", "evidence_ids": ["e1"]}],
                "medium": [{"finding": "Notable observation"}],
                "low": [{"finding": "Minor note"}],
            }
        )
        findings = extract_findings(synth, run_id="r1", timestamp="2025-01-01")
        assert len(findings) == 3
        assert findings[0].text == "Important result"
        assert findings[0].confidence == "high"
        assert findings[0].run_id == "r1"
        assert findings[0].evidence_ids == ["e1"]

    def test_string_findings(self) -> None:
        synth = _make_synthesis(
            findings={"high": ["finding as string"], "medium": [], "low": []}
        )
        findings = extract_findings(synth)
        assert len(findings) == 1
        assert findings[0].text == "finding as string"

    def test_themes_extracted(self) -> None:
        synth = _make_synthesis(
            themes=[
                {"description": "Theme about memory"},
                {"theme": "Theme about retrieval"},
            ]
        )
        findings = extract_findings(synth)
        assert len(findings) == 2

    def test_combined_sources(self) -> None:
        synth = _make_synthesis(
            findings={"high": [{"finding": "F1"}]},
            themes=[{"description": "T1"}],
        )
        findings = extract_findings(synth)
        assert len(findings) == 2


# ===========================================================================
# load_synthesis tests
# ===========================================================================


class TestLoadSynthesis:
    """Tests for load_synthesis."""

    def test_load_from_summarize_dir(self, tmp_path: Path) -> None:
        data = {"themes": [{"description": "test"}]}
        _write_synthesis(tmp_path, data)
        result = load_synthesis(tmp_path)
        assert result is not None
        assert result["themes"][0]["description"] == "test"

    def test_load_from_synthesis_dir(self, tmp_path: Path) -> None:
        synth_dir = tmp_path / "synthesis"
        synth_dir.mkdir()
        data = {"themes": []}
        (synth_dir / "synthesis_results.json").write_text(json.dumps(data))
        result = load_synthesis(tmp_path)
        assert result is not None

    def test_missing_synthesis(self, tmp_path: Path) -> None:
        result = load_synthesis(tmp_path)
        assert result is None

    def test_invalid_json(self, tmp_path: Path) -> None:
        synth_dir = tmp_path / "summarize"
        synth_dir.mkdir()
        (synth_dir / "synthesis_results.json").write_text("not json")
        result = load_synthesis(tmp_path)
        assert result is None


# ===========================================================================
# load_run_timestamp tests
# ===========================================================================


class TestLoadRunTimestamp:
    """Tests for load_run_timestamp."""

    def test_from_manifest(self, tmp_path: Path) -> None:
        _write_manifest(tmp_path, started_at="2025-06-15T10:00:00")
        ts = load_run_timestamp(tmp_path)
        assert ts == "2025-06-15T10:00:00"

    def test_fallback_to_mtime(self, tmp_path: Path) -> None:
        ts = load_run_timestamp(tmp_path)
        # Should return an ISO timestamp from mtime
        assert len(ts) > 0

    def test_invalid_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text("broken json")
        ts = load_run_timestamp(tmp_path)
        # Falls back to mtime
        assert len(ts) > 0


# ===========================================================================
# detect_contradictions tests
# ===========================================================================


class TestDetectContradictions:
    """Tests for detect_contradictions."""

    def test_no_contradictions(self) -> None:
        findings = {
            "r1": [Finding(text="transformers are effective for NLP tasks")],
            "r2": [Finding(text="transformers are effective for NLP tasks")],
        }
        contradictions = detect_contradictions(findings)
        assert contradictions == []

    def test_negation_contradiction(self) -> None:
        findings = {
            "r1": [
                Finding(text="memory systems improve agent performance significantly")
            ],
            "r2": [
                Finding(
                    text="memory systems do not improve agent performance significantly"
                )
            ],
        }
        contradictions = detect_contradictions(findings)
        assert len(contradictions) >= 1
        assert contradictions[0].similarity >= CONTRADICTION_SIMILARITY_THRESHOLD

    def test_unrelated_findings(self) -> None:
        findings = {
            "r1": [Finding(text="quantum computing advances")],
            "r2": [Finding(text="marine biology research methods")],
        }
        contradictions = detect_contradictions(findings)
        assert contradictions == []

    def test_single_run(self) -> None:
        findings = {
            "r1": [Finding(text="some finding")],
        }
        contradictions = detect_contradictions(findings)
        assert contradictions == []

    def test_empty_findings(self) -> None:
        contradictions = detect_contradictions({})
        assert contradictions == []

    def test_multiple_runs(self) -> None:
        findings = {
            "r1": [Finding(text="method A works well for retrieval tasks")],
            "r2": [Finding(text="method A works well for retrieval tasks")],
            "r3": [Finding(text="method A fails for retrieval tasks")],
        }
        contradictions = detect_contradictions(findings)
        # Should detect contradiction between r1/r2 and r3
        assert len(contradictions) >= 1


# ===========================================================================
# track_knowledge_updates tests
# ===========================================================================


class TestTrackKnowledgeUpdates:
    """Tests for track_knowledge_updates."""

    def test_new_findings(self) -> None:
        findings = {
            "r1": [],
            "r2": [Finding(text="brand new discovery")],
        }
        updates = track_knowledge_updates(findings, ["r1", "r2"])
        assert len(updates) == 1
        assert updates[0].update_type == "new"

    def test_retracted_findings(self) -> None:
        findings = {
            "r1": [Finding(text="old finding that disappears")],
            "r2": [],
        }
        updates = track_knowledge_updates(findings, ["r1", "r2"])
        assert len(updates) == 1
        assert updates[0].update_type == "retracted"

    def test_revised_findings(self) -> None:
        findings = {
            "r1": [
                Finding(text="transformers achieve good accuracy on benchmark tasks")
            ],
            "r2": [
                Finding(
                    text="transformers achieve state-of-the-art"
                    " accuracy on benchmark evaluation tasks"
                )
            ],
        }
        updates = track_knowledge_updates(findings, ["r1", "r2"])
        # Should find a revised update (similar but not identical)
        revised = [u for u in updates if u.update_type == "revised"]
        assert len(revised) >= 1

    def test_single_run(self) -> None:
        findings = {"r1": [Finding(text="finding")]}
        updates = track_knowledge_updates(findings, ["r1"])
        assert updates == []

    def test_empty_runs(self) -> None:
        findings = {"r1": [], "r2": []}
        updates = track_knowledge_updates(findings, ["r1", "r2"])
        assert updates == []

    def test_three_run_chain(self) -> None:
        findings = {
            "r1": [Finding(text="initial observation about memory")],
            "r2": [Finding(text="revised observation about memory systems")],
            "r3": [Finding(text="final observation about memory systems architecture")],
        }
        updates = track_knowledge_updates(findings, ["r1", "r2", "r3"])
        # Should have updates between r1→r2 and r2→r3
        assert len(updates) >= 2


# ===========================================================================
# Coherence dimension scoring tests
# ===========================================================================


class TestFactualConsistency:
    """Tests for compute_factual_consistency."""

    def test_perfect_consistency(self) -> None:
        findings = {
            "r1": [Finding(text="transformers are good for sequence modeling")],
            "r2": [Finding(text="transformers are good for sequence modeling tasks")],
        }
        score = compute_factual_consistency(findings)
        assert score == 1.0

    def test_single_run(self) -> None:
        assert compute_factual_consistency({"r1": [Finding(text="x")]}) == 1.0

    def test_no_topic_overlap(self) -> None:
        findings = {
            "r1": [Finding(text="quantum computing advances rapidly")],
            "r2": [Finding(text="marine biology discovers new species")],
        }
        score = compute_factual_consistency(findings)
        assert score == 1.0  # No overlap to be inconsistent about


class TestTemporalOrdering:
    """Tests for compute_temporal_ordering."""

    def test_single_run(self) -> None:
        assert compute_temporal_ordering({"r1": [Finding(text="x")]}, ["r1"]) == 1.0

    def test_clean_evolution(self) -> None:
        findings = {
            "r1": [],
            "r2": [Finding(text="new discovery in the field")],
        }
        score = compute_temporal_ordering(findings, ["r1", "r2"])
        assert score == 1.0  # All new findings, clean evolution


class TestKnowledgeUpdateFidelity:
    """Tests for compute_knowledge_update_fidelity."""

    def test_single_run(self) -> None:
        assert (
            compute_knowledge_update_fidelity({"r1": [Finding(text="x")]}, ["r1"])
            == 1.0
        )

    def test_all_new(self) -> None:
        findings = {
            "r1": [],
            "r2": [Finding(text="brand new finding")],
        }
        score = compute_knowledge_update_fidelity(findings, ["r1", "r2"])
        assert score == 1.0


class TestTopicOverlap:
    """Tests for compute_topic_overlap."""

    def test_no_overlap(self) -> None:
        findings = {
            "r1": [Finding(text="quantum physics breakthroughs")],
            "r2": [Finding(text="marine biology discoveries")],
        }
        overlap = compute_topic_overlap(findings)
        assert overlap == 0.0

    def test_single_run(self) -> None:
        assert compute_topic_overlap({"r1": [Finding(text="x")]}) == 0.0

    def test_full_overlap(self) -> None:
        findings = {
            "r1": [Finding(text="transformer architecture for NLP")],
            "r2": [Finding(text="transformer architecture for NLP tasks")],
        }
        overlap = compute_topic_overlap(findings)
        assert overlap > 0.0


# ===========================================================================
# evaluate_coherence integration tests
# ===========================================================================


class TestEvaluateCoherence:
    """Tests for evaluate_coherence."""

    def test_two_runs_with_synthesis(self, tmp_path: Path) -> None:
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _write_manifest(run1, "2025-01-01T00:00:00")
        _write_manifest(run2, "2025-02-01T00:00:00")
        _write_synthesis(
            run1,
            _make_synthesis(
                findings={"high": [{"finding": "Method A improves accuracy"}]}
            ),
        )
        _write_synthesis(
            run2,
            _make_synthesis(
                findings={
                    "high": [{"finding": "Method A improves accuracy significantly"}]
                }
            ),
        )

        report = evaluate_coherence([run1, run2], ["run1", "run2"])
        assert isinstance(report, CoherenceReport)
        assert report.score.overall >= 0.0
        assert report.score.overall <= 1.0
        assert len(report.run_ids) == 2
        assert report.finding_count >= 2

    def test_mismatched_lengths(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="same length"):
            evaluate_coherence([tmp_path], ["r1", "r2"])

    def test_no_synthesis(self, tmp_path: Path) -> None:
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        run1.mkdir()
        run2.mkdir()
        report = evaluate_coherence([run1, run2], ["run1", "run2"])
        assert report.finding_count == 0
        assert report.score.factual_consistency == 1.0

    def test_custom_weights(self, tmp_path: Path) -> None:
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _write_manifest(run1, "2025-01-01T00:00:00")
        _write_manifest(run2, "2025-02-01T00:00:00")
        _write_synthesis(run1, _make_synthesis(findings={"high": [{"finding": "F1"}]}))
        _write_synthesis(run2, _make_synthesis(findings={"high": [{"finding": "F2"}]}))

        weights = {
            "factual_consistency": 0.5,
            "temporal_ordering": 0.2,
            "knowledge_update_fidelity": 0.2,
            "contradiction_rate": 0.1,
        }
        report = evaluate_coherence([run1, run2], ["run1", "run2"], weights=weights)
        assert report.score.overall >= 0.0

    def test_three_runs(self, tmp_path: Path) -> None:
        runs = []
        rids = []
        for i in range(3):
            r = tmp_path / f"run{i}"
            _write_manifest(r, f"2025-0{i + 1}-01T00:00:00")
            _write_synthesis(
                r,
                _make_synthesis(
                    findings={"high": [{"finding": f"Finding {i} about topic"}]}
                ),
            )
            runs.append(r)
            rids.append(f"run{i}")

        report = evaluate_coherence(runs, rids)
        assert len(report.run_ids) == 3
        assert report.finding_count == 3


# ===========================================================================
# run_coherence end-to-end tests
# ===========================================================================


class TestRunCoherence:
    """Tests for run_coherence."""

    def test_end_to_end(self, tmp_path: Path) -> None:
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _write_manifest(run1, "2025-01-01T00:00:00")
        _write_manifest(run2, "2025-02-01T00:00:00")
        _write_synthesis(
            run1, _make_synthesis(findings={"high": [{"finding": "Result A"}]})
        )
        _write_synthesis(
            run2, _make_synthesis(findings={"high": [{"finding": "Result B"}]})
        )

        report = run_coherence(
            run_ids=["run1", "run2"],
            workspace=tmp_path,
        )
        assert isinstance(report, CoherenceReport)
        # Check output file was written
        out_files = list(tmp_path.glob("coherence_*.json"))
        assert len(out_files) == 1

    def test_custom_output(self, tmp_path: Path) -> None:
        run1 = tmp_path / "run1"
        run2 = tmp_path / "run2"
        _write_manifest(run1, "2025-01-01T00:00:00")
        _write_manifest(run2, "2025-02-01T00:00:00")
        _write_synthesis(run1, _make_synthesis())
        _write_synthesis(run2, _make_synthesis())

        out = tmp_path / "output" / "report.json"
        run_coherence(
            run_ids=["run1", "run2"],
            workspace=tmp_path,
            output=out,
        )
        assert out.exists()

    def test_too_few_runs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="At least 2"):
            run_coherence(run_ids=["r1"], workspace=tmp_path)

    def test_missing_run_dir(self, tmp_path: Path) -> None:
        (tmp_path / "run1").mkdir()
        with pytest.raises(FileNotFoundError):
            run_coherence(
                run_ids=["run1", "nonexistent"],
                workspace=tmp_path,
            )


# ===========================================================================
# Threshold constant tests
# ===========================================================================


class TestThresholds:
    """Verify threshold constants are reasonable."""

    def test_topic_threshold(self) -> None:
        assert 0.0 < TOPIC_SIMILARITY_THRESHOLD < 1.0

    def test_contradiction_threshold(self) -> None:
        assert 0.0 < CONTRADICTION_SIMILARITY_THRESHOLD < 1.0

    def test_contradiction_ge_topic(self) -> None:
        assert CONTRADICTION_SIMILARITY_THRESHOLD >= TOPIC_SIMILARITY_THRESHOLD
