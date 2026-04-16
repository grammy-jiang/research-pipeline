"""Tests for epistemic blinding audits (B5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_pipeline.evaluation.blinding import (
    FEATURE_TYPES,
    HIGH_CONTAMINATION_THRESHOLD,
    MEDIUM_CONTAMINATION_THRESHOLD,
    AuditStore,
    BlindedDocument,
    BlindingAuditResult,
    BlindingMask,
    ContaminationScore,
    IdentifyingFeature,
    audit_paper_summary,
    blind_document,
    detect_identifying_features,
    run_blinding_audit,
    run_blinding_audit_for_workspace,
    score_contamination,
)

# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestBlindingMask:
    """Tests for BlindingMask dataclass."""

    def test_default_all_active(self) -> None:
        mask = BlindingMask()
        assert mask.active_features() == list(FEATURE_TYPES)

    def test_partial_mask(self) -> None:
        mask = BlindingMask(
            authors=True, title=False, venue=False, year=True, citations=False
        )
        assert mask.active_features() == ["authors", "year"]

    def test_empty_mask(self) -> None:
        mask = BlindingMask(
            authors=False, title=False, venue=False, year=False, citations=False
        )
        assert mask.active_features() == []


class TestIdentifyingFeature:
    """Tests for IdentifyingFeature dataclass."""

    def test_creation(self) -> None:
        feat = IdentifyingFeature(
            feature_type="authors", value="Smith", locations=[10, 50]
        )
        assert feat.feature_type == "authors"
        assert feat.value == "Smith"
        assert feat.locations == [10, 50]

    def test_default_locations(self) -> None:
        feat = IdentifyingFeature(feature_type="venue", value="NeurIPS")
        assert feat.locations == []


class TestBlindedDocument:
    """Tests for BlindedDocument dataclass."""

    def test_mask_count(self) -> None:
        doc = BlindedDocument(
            original_text="hello",
            blinded_text="hello",
            features_masked=[
                IdentifyingFeature("authors", "A", [1, 2]),
                IdentifyingFeature("venue", "B", [3]),
            ],
        )
        assert doc.mask_count == 3

    def test_empty_mask_count(self) -> None:
        doc = BlindedDocument(original_text="x", blinded_text="x")
        assert doc.mask_count == 0


class TestContaminationScore:
    """Tests for ContaminationScore dataclass."""

    def test_contamination_ratio(self) -> None:
        score = ContaminationScore(
            paper_id="abc",
            total_claims=10,
            contaminated_claims=3,
        )
        assert score.contamination_ratio == pytest.approx(0.3)

    def test_zero_claims_ratio(self) -> None:
        score = ContaminationScore(
            paper_id="abc", total_claims=0, contaminated_claims=0
        )
        assert score.contamination_ratio == 0.0


class TestBlindingAuditResult:
    """Tests for BlindingAuditResult dataclass."""

    def test_to_dict_roundtrip(self) -> None:
        result = BlindingAuditResult(
            run_id="run-001",
            timestamp="2025-01-01T00:00:00Z",
            paper_scores=[
                ContaminationScore(
                    paper_id="p1",
                    feature_scores={"authors": 0.5, "title": 0.2},
                    overall_score=0.35,
                    identity_references=3,
                    total_claims=10,
                    contaminated_claims=2,
                )
            ],
            aggregate_score=0.35,
            high_contamination_papers=["p1"],
            recommendation="HIGH contamination",
        )
        d = result.to_dict()
        assert d["run_id"] == "run-001"
        assert len(d["paper_scores"]) == 1
        assert d["paper_scores"][0]["paper_id"] == "p1"
        assert d["paper_scores"][0]["contamination_ratio"] == pytest.approx(0.2)
        assert d["high_contamination_papers"] == ["p1"]


# ---------------------------------------------------------------------------
# Feature detection tests
# ---------------------------------------------------------------------------


class TestDetectIdentifyingFeatures:
    """Tests for detect_identifying_features."""

    def test_detect_author_full_name(self) -> None:
        text = "This work by John Smith introduces a novel approach."
        features = detect_identifying_features(text, authors=["John Smith"])
        author_feats = [f for f in features if f.feature_type == "authors"]
        assert len(author_feats) >= 1
        values = [f.value for f in author_feats]
        assert "John Smith" in values

    def test_detect_author_last_name(self) -> None:
        text = "Smith et al. propose a method for text classification."
        features = detect_identifying_features(text, authors=["John Smith"])
        author_feats = [f for f in features if f.feature_type == "authors"]
        # Should find "Smith" as last name
        last_name_feats = [f for f in author_feats if f.value == "Smith"]
        assert len(last_name_feats) >= 1

    def test_detect_title_full(self) -> None:
        text = "The paper Attention Is All You Need revolutionized NLP."
        features = detect_identifying_features(text, title="Attention Is All You Need")
        title_feats = [f for f in features if f.feature_type == "title"]
        assert len(title_feats) >= 1

    def test_detect_title_fragment(self) -> None:
        text = "The concept of Attention Is All You Need introduced transformers."
        features = detect_identifying_features(
            text, title="Attention Is All You Need For Modern Transformers"
        )
        title_feats = [f for f in features if f.feature_type == "title"]
        assert len(title_feats) >= 1

    def test_detect_venue_known_pattern(self) -> None:
        text = "Published at NeurIPS 2023."
        features = detect_identifying_features(text)
        venue_feats = [f for f in features if f.feature_type == "venue"]
        assert any(f.value == "NeurIPS" for f in venue_feats)

    def test_detect_venue_explicit(self) -> None:
        text = "Accepted at the International Workshop on ML."
        features = detect_identifying_features(
            text, venue="International Workshop on ML"
        )
        venue_feats = [f for f in features if f.feature_type == "venue"]
        assert len(venue_feats) >= 1

    def test_detect_year(self) -> None:
        text = "This 2023 paper builds on prior work from 2021."
        features = detect_identifying_features(text, year=2023)
        year_feats = [f for f in features if f.feature_type == "year"]
        assert any(f.value == "2023" for f in year_feats)

    def test_detect_citations_bracket(self) -> None:
        text = "As shown in [1], the method improves accuracy [2,3]."
        features = detect_identifying_features(text)
        cite_feats = [f for f in features if f.feature_type == "citations"]
        assert len(cite_feats) >= 2

    def test_detect_citations_author_year(self) -> None:
        text = "Following (Vaswani et al., 2017), we propose..."
        features = detect_identifying_features(text)
        cite_feats = [f for f in features if f.feature_type == "citations"]
        assert len(cite_feats) >= 1

    def test_mask_filter(self) -> None:
        text = "Smith published at NeurIPS 2023."
        mask = BlindingMask(
            authors=True, venue=False, year=False, citations=False, title=False
        )
        features = detect_identifying_features(text, authors=["John Smith"], mask=mask)
        # Should only find author features
        types = {f.feature_type for f in features}
        assert "venue" not in types

    def test_no_features_in_clean_text(self) -> None:
        text = "The method uses gradient descent to optimize the loss function."
        features = detect_identifying_features(text)
        # Only venue patterns might match, but this text has none
        assert all(f.feature_type == "venue" for f in features) or len(features) == 0

    def test_short_author_name_skipped(self) -> None:
        text = "Li proposed a method for classification."
        features = detect_identifying_features(text, authors=["Li"])
        # "Li" is < 3 chars, should be skipped
        author_feats = [f for f in features if f.feature_type == "authors"]
        assert len(author_feats) == 0


# ---------------------------------------------------------------------------
# Document blinding tests
# ---------------------------------------------------------------------------


class TestBlindDocument:
    """Tests for blind_document."""

    def test_blind_author(self) -> None:
        text = "Smith et al. propose a transformer model."
        doc = blind_document(text, authors=["John Smith"])
        assert "[MASKED_AUTHORS]" in doc.blinded_text
        assert "Smith" not in doc.blinded_text.replace("[MASKED_AUTHORS]", "")

    def test_blind_venue(self) -> None:
        text = "Published at NeurIPS with great results."
        doc = blind_document(text)
        assert "[MASKED_VENUE]" in doc.blinded_text

    def test_blind_year(self) -> None:
        text = "In 2023, a new approach was proposed."
        doc = blind_document(text, year=2023)
        assert "[MASKED_YEAR]" in doc.blinded_text

    def test_blind_preserves_original(self) -> None:
        text = "Smith published at NeurIPS."
        doc = blind_document(text, authors=["John Smith"])
        assert doc.original_text == text
        assert doc.blinded_text != text

    def test_no_mask_returns_original(self) -> None:
        text = "A method for classification."
        mask = BlindingMask(
            authors=False, title=False, venue=False, year=False, citations=False
        )
        doc = blind_document(text, mask=mask)
        assert doc.blinded_text == text

    def test_multiple_features_masked(self) -> None:
        text = "Smith from NeurIPS 2023 found [1] that attention works."
        doc = blind_document(text, authors=["John Smith"], year=2023)
        assert "[MASKED_AUTHORS]" in doc.blinded_text
        assert "[MASKED_VENUE]" in doc.blinded_text
        assert "[MASKED_YEAR]" in doc.blinded_text
        assert "[MASKED_CITATIONS]" in doc.blinded_text


# ---------------------------------------------------------------------------
# Contamination scoring tests
# ---------------------------------------------------------------------------


class TestScoreContamination:
    """Tests for score_contamination."""

    def test_clean_findings(self) -> None:
        findings = [
            "The method achieves 95% accuracy on the benchmark.",
            "Gradient descent converges in 100 epochs.",
            "The loss function decreases monotonically.",
        ]
        score = score_contamination(findings, authors=["Jane Doe"], title="My Paper")
        assert score.overall_score < MEDIUM_CONTAMINATION_THRESHOLD
        assert score.contaminated_claims == 0

    def test_contaminated_by_author(self) -> None:
        findings = [
            "Building on Smith's prior work, the method improves accuracy.",
            "Smith et al. showed that transformers outperform LSTMs.",
            "The gradient descent optimization converges quickly.",
        ]
        score = score_contamination(findings, authors=["John Smith"])
        assert score.feature_scores["authors"] > 0
        assert score.contaminated_claims >= 2

    def test_contaminated_by_venue(self) -> None:
        findings = [
            "As shown at NeurIPS, this approach is effective.",
            "The ICML benchmark demonstrates strong results.",
        ]
        score = score_contamination(findings)
        assert score.feature_scores["venue"] > 0

    def test_empty_findings(self) -> None:
        score = score_contamination([], authors=["Smith"])
        assert score.overall_score == 0.0
        assert score.total_claims == 0

    def test_contaminated_by_year(self) -> None:
        findings = [
            "The 2023 result shows improvement.",
            "Compared to the 2023 baseline, accuracy increased.",
        ]
        score = score_contamination(findings, year=2023)
        assert score.feature_scores["year"] > 0

    def test_contaminated_by_citation(self) -> None:
        findings = [
            "As shown in [1], the method works.",
            "Following [2,3], we improved the model.",
        ]
        score = score_contamination(findings)
        assert score.feature_scores["citations"] > 0

    def test_heavy_contamination_high_score(self) -> None:
        findings = [
            "Smith (NeurIPS 2023) showed attention works [1].",
            "Building on Smith's NeurIPS work from 2023 [2].",
            "Smith et al., NeurIPS 2023 demonstrated [3].",
        ]
        score = score_contamination(
            findings, authors=["John Smith"], venue="NeurIPS", year=2023
        )
        assert score.overall_score > MEDIUM_CONTAMINATION_THRESHOLD
        assert score.contamination_ratio > 0.5

    def test_score_capped_at_one(self) -> None:
        findings = ["Smith Smith Smith Smith"] * 5
        score = score_contamination(findings, authors=["John Smith"])
        for v in score.feature_scores.values():
            assert v <= 1.0


# ---------------------------------------------------------------------------
# Paper summary audit tests
# ---------------------------------------------------------------------------


class TestAuditPaperSummary:
    """Tests for audit_paper_summary."""

    def test_clean_summary(self) -> None:
        summary = {
            "title": "A Novel Method for Classification",
            "authors": ["Jane Doe"],
            "findings": [
                "Accuracy improved by 5% on benchmark datasets.",
                "The model converges in fewer epochs than baseline.",
            ],
        }
        score = audit_paper_summary(summary)
        assert score.overall_score < HIGH_CONTAMINATION_THRESHOLD

    def test_contaminated_summary(self) -> None:
        summary = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "venue": "NeurIPS",
            "year": 2017,
            "findings": [
                "Vaswani introduced the transformer at NeurIPS 2017.",
                "The Attention Is All You Need paper changed NLP [1].",
            ],
            "methodology": "Following Vaswani's approach from NeurIPS.",
            "objective": "Extending the 2017 Vaswani transformer.",
        }
        score = audit_paper_summary(summary)
        assert score.overall_score > MEDIUM_CONTAMINATION_THRESHOLD
        assert score.contaminated_claims > 0

    def test_includes_methodology_and_objective(self) -> None:
        summary = {
            "title": "Test Paper",
            "authors": ["Bob"],
            "findings": [],
            "methodology": "A standard approach.",
            "objective": "To improve accuracy.",
        }
        score = audit_paper_summary(summary)
        # 2 claims: methodology + objective
        assert score.total_claims == 2


# ---------------------------------------------------------------------------
# Run-level audit tests
# ---------------------------------------------------------------------------


class TestRunBlindingAudit:
    """Tests for run_blinding_audit."""

    def test_audit_with_synthesis_report(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-001"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Transformer Methods",
                    "authors": ["Alice Chen"],
                    "findings": [
                        "The model achieves state-of-the-art results.",
                        "Training converges in 50 epochs.",
                    ],
                }
            ],
            "agreements": [{"claim": "Transformers are effective."}],
            "disagreements": [],
            "gaps": [{"description": "More data needed."}],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = run_blinding_audit(run_dir, "run-001")
        assert result.run_id == "run-001"
        assert len(result.paper_scores) >= 1
        assert result.recommendation != ""

    def test_audit_with_individual_summaries(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-002"
        summaries_dir = run_dir / "summarize" / "summaries"
        summaries_dir.mkdir(parents=True)

        summary = {
            "title": "Deep Learning for NLP",
            "authors": ["Bob Smith"],
            "findings": ["Accuracy improved to 98%."],
        }
        (summaries_dir / "paper1.json").write_text(json.dumps(summary))

        result = run_blinding_audit(run_dir, "run-002")
        assert len(result.paper_scores) >= 1

    def test_audit_empty_run(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-003"
        run_dir.mkdir(parents=True)
        (run_dir / "summarize").mkdir()

        result = run_blinding_audit(run_dir, "run-003")
        assert len(result.paper_scores) == 0
        assert "No paper summaries" in result.recommendation

    def test_audit_with_screened_candidates(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-004"
        screen_dir = run_dir / "screen"
        screen_dir.mkdir(parents=True)
        (run_dir / "summarize").mkdir(parents=True)

        candidates = [
            {
                "title": "Test Paper",
                "authors": ["Smith"],
                "rationale": "Smith's work at NeurIPS is highly relevant.",
            }
        ]
        (screen_dir / "screened.json").write_text(json.dumps(candidates))

        result = run_blinding_audit(run_dir, "run-004")
        assert len(result.paper_scores) >= 1


# ---------------------------------------------------------------------------
# AuditStore tests
# ---------------------------------------------------------------------------


class TestAuditStore:
    """Tests for AuditStore SQLite storage."""

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_audit.db"
        store = AuditStore(db_path)
        try:
            result = BlindingAuditResult(
                run_id="run-100",
                timestamp="2025-01-01T00:00:00Z",
                paper_scores=[
                    ContaminationScore(
                        paper_id="p1",
                        feature_scores={"authors": 0.5},
                        overall_score=0.3,
                        identity_references=2,
                        total_claims=5,
                        contaminated_claims=2,
                    )
                ],
                aggregate_score=0.3,
                high_contamination_papers=[],
                recommendation="LOW contamination.",
            )
            audit_id = store.store_audit(result)
            assert audit_id >= 1

            retrieved = store.get_audit(audit_id)
            assert retrieved is not None
            assert retrieved.run_id == "run-100"
            assert len(retrieved.paper_scores) == 1
            assert retrieved.paper_scores[0].paper_id == "p1"
        finally:
            store.close()

    def test_get_nonexistent_audit(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_audit2.db"
        store = AuditStore(db_path)
        try:
            assert store.get_audit(999) is None
        finally:
            store.close()

    def test_get_audits_for_run(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_audit3.db"
        store = AuditStore(db_path)
        try:
            for i in range(3):
                result = BlindingAuditResult(
                    run_id="run-200",
                    timestamp=f"2025-01-0{i + 1}T00:00:00Z",
                    aggregate_score=0.1 * (i + 1),
                    recommendation=f"Audit {i + 1}",
                )
                store.store_audit(result)

            audits = store.get_audits_for_run("run-200")
            assert len(audits) == 3
            # Should be ordered by timestamp DESC
            assert audits[0].timestamp > audits[1].timestamp
        finally:
            store.close()

    def test_get_paper_history(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_audit4.db"
        store = AuditStore(db_path)
        try:
            for i in range(2):
                result = BlindingAuditResult(
                    run_id=f"run-30{i}",
                    timestamp=f"2025-01-0{i + 1}T00:00:00Z",
                    paper_scores=[
                        ContaminationScore(
                            paper_id="paper-abc",
                            overall_score=0.2 * (i + 1),
                            identity_references=i + 1,
                            total_claims=5,
                            contaminated_claims=i,
                        )
                    ],
                    aggregate_score=0.2 * (i + 1),
                    recommendation="test",
                )
                store.store_audit(result)

            history = store.get_paper_history("paper-abc")
            assert len(history) == 2
            assert history[0]["run_id"] == "run-301"  # most recent first
        finally:
            store.close()

    def test_get_all_audits(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_audit5.db"
        store = AuditStore(db_path)
        try:
            for i in range(2):
                store.store_audit(
                    BlindingAuditResult(
                        run_id=f"run-{i}",
                        timestamp=f"2025-0{i + 1}-01T00:00:00Z",
                        aggregate_score=0.1,
                        recommendation="test",
                    )
                )
            all_audits = store.get_all_audits()
            assert len(all_audits) == 2
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Workspace-level audit tests
# ---------------------------------------------------------------------------


class TestRunBlindingAuditForWorkspace:
    """Tests for run_blinding_audit_for_workspace."""

    def test_audit_workspace_latest_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        runs_dir = workspace / "runs"
        run_dir = runs_dir / "20250101_120000"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Test Paper",
                    "authors": ["Author A"],
                    "findings": ["The method works well on benchmarks."],
                }
            ],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = run_blinding_audit_for_workspace(workspace, store_results=True)
        assert result.run_id == "20250101_120000"

        # Check DB was created
        db_path = workspace / ".blinding_audits.db"
        assert db_path.exists()

    def test_audit_specific_run(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        run_dir = workspace / "runs" / "my-run"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = run_blinding_audit_for_workspace(
            workspace, run_id="my-run", store_results=False
        )
        assert result.run_id == "my-run"

    def test_audit_no_runs_dir(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        result = run_blinding_audit_for_workspace(workspace)
        assert "No runs directory" in result.recommendation

    def test_audit_empty_runs(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "runs").mkdir(parents=True)
        result = run_blinding_audit_for_workspace(workspace)
        assert "No runs found" in result.recommendation

    def test_audit_nonexistent_run_id(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "runs").mkdir(parents=True)
        result = run_blinding_audit_for_workspace(workspace, run_id="missing")
        assert "not found" in result.recommendation


# ---------------------------------------------------------------------------
# Recommendation generation tests
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for recommendation generation logic."""

    def test_low_contamination_recommendation(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Clean Paper",
                    "authors": ["Unknown"],
                    "findings": [
                        "The gradient converges monotonically.",
                        "Accuracy reaches 95% on test set.",
                    ],
                }
            ],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = run_blinding_audit(run_dir, "run-clean")
        assert "LOW" in result.recommendation

    def test_high_contamination_recommendation(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        summarize_dir = run_dir / "summarize"
        summarize_dir.mkdir(parents=True)

        synthesis = {
            "per_paper_summaries": [
                {
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "venue": "NeurIPS",
                    "year": 2017,
                    "findings": [
                        "Vaswani introduced transformers at NeurIPS 2017 [1].",
                        "Shazeer co-authored the NeurIPS 2017 breakthrough [2].",
                        "The Attention Is All You Need paper from NeurIPS 2017.",
                    ],
                    "methodology": "Vaswani's NeurIPS transformer approach.",
                    "objective": "Extending the 2017 Vaswani NeurIPS work.",
                }
            ],
            "agreements": [],
            "disagreements": [],
            "gaps": [],
        }
        (summarize_dir / "synthesis_report.json").write_text(json.dumps(synthesis))

        result = run_blinding_audit(run_dir, "run-dirty", contamination_threshold=0.2)
        assert result.aggregate_score > MEDIUM_CONTAMINATION_THRESHOLD
