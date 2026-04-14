"""Tests for claim decomposition and evidence taxonomy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.analysis.decomposer import (
    _classify_evidence_heuristic,
    _split_into_atomic,
    decompose_paper,
)
from research_pipeline.models.claim import (
    AtomicClaim,
    ClaimDecomposition,
    ClaimEvidence,
    EvidenceClass,
)
from research_pipeline.models.extraction import ChunkMetadata
from research_pipeline.models.summary import PaperSummary

# ── Fixtures ──────────────────────────────────────────────────────────


def _make_summary(**overrides: object) -> PaperSummary:
    """Create a realistic PaperSummary for testing."""
    defaults = {
        "arxiv_id": "2401.12345",
        "version": "v1",
        "title": "Attention Is All You Need: Revisited",
        "objective": "This paper proposes a novel transformer architecture.",
        "methodology": "We train models on WMT 2014 English-to-German.",
        "findings": [
            "The model achieves 28.4 BLEU on WMT 2014;"
            " This is 2 points above the baseline.",
            "Training time is reduced by 50% compared to recurrent models.",
        ],
        "limitations": [
            "The model struggles with very long sequences,"
            " while it excels at shorter ones.",
        ],
    }
    defaults.update(overrides)
    return PaperSummary(**defaults)


def _make_chunk(
    chunk_id: str, text: str, paper_id: str = "2401.12345"
) -> tuple[ChunkMetadata, str]:
    """Create a (ChunkMetadata, text) pair."""
    meta = ChunkMetadata(
        paper_id=paper_id,
        section_path="Results",
        chunk_id=chunk_id,
        source_span="L10-L20",
        token_count=50,
    )
    return (meta, text)


def _make_scored_chunk(
    chunk_id: str, text: str, score: float, paper_id: str = "2401.12345"
) -> tuple[ChunkMetadata, str, float]:
    """Create a scored chunk tuple for evidence classification."""
    meta = ChunkMetadata(
        paper_id=paper_id,
        section_path="Results",
        chunk_id=chunk_id,
        source_span="L10-L20",
        token_count=50,
    )
    return (meta, text, score)


# ── _split_into_atomic tests ─────────────────────────────────────────


class TestSplitIntoAtomic:
    """Tests for the _split_into_atomic heuristic splitter."""

    def test_simple_sentence_returns_as_is(self) -> None:
        result = _split_into_atomic("The model achieves state-of-the-art results.")
        assert result == ["The model achieves state-of-the-art results."]

    def test_semicolon_splits(self) -> None:
        text = "The model is fast; The accuracy is high; Memory usage is low."
        result = _split_into_atomic(text)
        assert len(result) == 3
        assert "The model is fast" in result
        assert "The accuracy is high" in result
        assert "Memory usage is low." in result

    def test_and_with_uppercase_splits(self) -> None:
        text = "The model is fast, and The accuracy improves significantly."
        result = _split_into_atomic(text)
        assert len(result) == 2
        assert "The model is fast" in result
        assert "The accuracy improves significantly." in result

    def test_and_lowercase_does_not_split(self) -> None:
        text = "The model uses attention and normalization layers."
        result = _split_into_atomic(text)
        assert len(result) == 1

    def test_while_contrast_splits(self) -> None:
        text = "The model excels at short sequences, while it struggles with long ones."
        result = _split_into_atomic(text)
        assert len(result) == 2
        assert "The model excels at short sequences" in result
        assert "it struggles with long ones." in result

    def test_whereas_contrast_splits(self) -> None:
        text = "Method A is faster, whereas Method B is more accurate overall."
        result = _split_into_atomic(text)
        assert len(result) == 2

    def test_empty_string_filtered(self) -> None:
        result = _split_into_atomic("")
        assert result == []

    def test_short_strings_filtered(self) -> None:
        result = _split_into_atomic("OK; Yes; No")
        assert result == []

    def test_complex_multi_split(self) -> None:
        text = (
            "The model is efficient; It achieves 95% accuracy, "
            "and The training is stable, while the inference is fast."
        )
        result = _split_into_atomic(text)
        assert len(result) >= 3

    def test_whitespace_handling(self) -> None:
        text = "  First claim here  ;  Second claim here  "
        result = _split_into_atomic(text)
        assert all(s == s.strip() for s in result)
        assert len(result) == 2


# ── _classify_evidence_heuristic tests ────────────────────────────────


class TestClassifyEvidenceHeuristic:
    """Tests for the heuristic evidence classifier."""

    def test_high_score_returns_supported(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "The model achieves excellent results.", 0.5),
            _make_scored_chunk("c2", "Performance is good.", 0.2),
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic(
            "model achieves", chunks
        )
        assert ev_class == EvidenceClass.SUPPORTED
        assert len(evidence) == 2
        assert conf > 0.0

    def test_medium_score_returns_partial(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "Some results suggest improvement.", 0.2),
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic(
            "model improves", chunks
        )
        assert ev_class == EvidenceClass.PARTIAL
        assert conf == 0.2

    def test_very_low_score_returns_unsupported(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "Unrelated content here.", 0.02),
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic(
            "specific claim", chunks
        )
        assert ev_class == EvidenceClass.UNSUPPORTED
        assert conf == 0.0

    def test_empty_chunks_returns_unsupported(self) -> None:
        ev_class, evidence, conf = _classify_evidence_heuristic("any claim", [])
        assert ev_class == EvidenceClass.UNSUPPORTED
        assert evidence == []
        assert conf == 0.0

    def test_negation_in_high_score_returns_conflicting(self) -> None:
        chunks = [
            _make_scored_chunk(
                "c1",
                "However, the model fails to generalize beyond training data.",
                0.4,
            ),
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic(
            "model generalizes", chunks
        )
        assert ev_class == EvidenceClass.CONFLICTING
        assert conf > 0.0

    def test_ambiguous_score_returns_inconclusive(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "Results are mixed and need more study.", 0.08),
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic("results mixed", chunks)
        assert ev_class == EvidenceClass.INCONCLUSIVE
        assert conf > 0.0

    def test_top_5_evidence_limit(self) -> None:
        chunks = [
            _make_scored_chunk(f"c{i}", f"Content chunk {i}", 0.5 - i * 0.05)
            for i in range(10)
        ]
        ev_class, evidence, conf = _classify_evidence_heuristic("content", chunks)
        assert len(evidence) <= 5

    def test_confidence_capped_at_one(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "Strong match.", 0.95),
        ]
        _, _, conf = _classify_evidence_heuristic("strong match", chunks)
        assert conf <= 1.0

    def test_custom_thresholds(self) -> None:
        chunks = [
            _make_scored_chunk("c1", "Some content.", 0.25),
        ]
        ev_class, _, _ = _classify_evidence_heuristic(
            "claim",
            chunks,
            threshold_supported=0.5,
            threshold_partial=0.2,
        )
        assert ev_class == EvidenceClass.PARTIAL


# ── decompose_paper tests ─────────────────────────────────────────────


class TestDecomposePaper:
    """Tests for the full decompose_paper function."""

    def test_basic_decomposition_no_chunks(self) -> None:
        summary = _make_summary()
        result = decompose_paper(summary)

        assert isinstance(result, ClaimDecomposition)
        assert result.paper_id == "2401.12345"
        assert result.title == "Attention Is All You Need: Revisited"
        assert result.total_claims > 0
        assert result.total_claims == len(result.claims)
        # All claims should be UNSUPPORTED without chunks
        for claim in result.claims:
            assert claim.evidence_class == EvidenceClass.UNSUPPORTED

    def test_claim_ids_sequential(self) -> None:
        summary = _make_summary()
        result = decompose_paper(summary)

        for i, claim in enumerate(result.claims, 1):
            assert claim.claim_id == f"CL-{i:03d}"

    def test_source_types_present(self) -> None:
        summary = _make_summary()
        result = decompose_paper(summary)

        source_types = {c.source_type for c in result.claims}
        assert "objective" in source_types
        assert "methodology" in source_types
        assert "finding" in source_types
        assert "limitation" in source_types

    def test_paper_id_propagated(self) -> None:
        summary = _make_summary(arxiv_id="9999.00001")
        result = decompose_paper(summary)

        for claim in result.claims:
            assert claim.paper_id == "9999.00001"

    def test_evidence_summary_counts_match(self) -> None:
        summary = _make_summary()
        result = decompose_paper(summary)

        counted: dict[str, int] = {}
        for claim in result.claims:
            key = claim.evidence_class.value
            counted[key] = counted.get(key, 0) + 1
        assert result.evidence_summary == counted

    def test_empty_findings_handled(self) -> None:
        summary = _make_summary(findings=[], limitations=[])
        result = decompose_paper(summary)

        # Should still have objective and methodology claims
        assert result.total_claims > 0
        source_types = {c.source_type for c in result.claims}
        assert "objective" in source_types
        assert "methodology" in source_types

    def test_empty_all_fields_handled(self) -> None:
        summary = _make_summary(
            objective="",
            methodology="",
            findings=[],
            limitations=[],
        )
        result = decompose_paper(summary)
        assert result.total_claims == 0
        assert result.claims == []

    def test_with_precomputed_chunks(self) -> None:
        summary = _make_summary(
            findings=["The transformer model achieves state-of-the-art results."]
        )
        chunks = [
            _make_chunk(
                "c1",
                "The transformer model achieves state-of-the-art"
                " results on benchmarks.",
            ),
            _make_chunk("c2", "Our architecture uses self-attention mechanisms."),
        ]

        with patch(
            "research_pipeline.analysis.decomposer.retrieve_relevant_chunks"
        ) as mock_retrieve:
            mock_retrieve.return_value = [
                (chunks[0][0], chunks[0][1], 0.6),
                (chunks[1][0], chunks[1][1], 0.1),
            ]
            result = decompose_paper(summary, chunks=chunks)

        # At least some claims should have evidence
        has_evidence = any(
            c.evidence_class != EvidenceClass.UNSUPPORTED for c in result.claims
        )
        assert has_evidence

    def test_with_markdown_path(self) -> None:
        summary = _make_summary(findings=["The approach reduces training time by 50%."])
        md_content = (
            "# Results\n\nThe approach reduces training time by 50% on all datasets."
        )

        with (
            patch("research_pipeline.analysis.decomposer.chunk_markdown") as mock_chunk,
            patch(
                "research_pipeline.analysis.decomposer.retrieve_relevant_chunks"
            ) as mock_retrieve,
            patch("pathlib.Path.read_text", return_value=md_content),
        ):
            fake_meta = ChunkMetadata(
                paper_id="2401.12345",
                section_path="Results",
                chunk_id="c1",
                source_span="L1-L3",
                token_count=20,
            )
            mock_chunk.return_value = [(fake_meta, md_content)]
            mock_retrieve.return_value = [(fake_meta, md_content, 0.5)]

            result = decompose_paper(summary, markdown_path="/fake/path.md")

        assert result.total_claims > 0

    def test_semicolon_in_finding_produces_multiple_claims(self) -> None:
        summary = _make_summary(
            findings=[
                "BLEU score reaches 28.4;"
                " This outperforms the previous best by 2 points."
            ],
            limitations=[],
        )
        result = decompose_paper(summary)

        finding_claims = [c for c in result.claims if c.source_type == "finding"]
        assert len(finding_claims) >= 2

    def test_llm_provider_accepted_but_unused(self) -> None:
        summary = _make_summary()
        mock_llm = MagicMock()
        result = decompose_paper(summary, llm_provider=mock_llm)
        assert result.total_claims > 0
        # LLM not called in heuristic mode
        mock_llm.call.assert_not_called()


# ── Model roundtrip tests ─────────────────────────────────────────────


class TestModels:
    """Tests for Pydantic model serialization."""

    def test_evidence_class_values(self) -> None:
        assert EvidenceClass.SUPPORTED.value == "supported"
        assert EvidenceClass.PARTIAL.value == "partial"
        assert EvidenceClass.CONFLICTING.value == "conflicting"
        assert EvidenceClass.INCONCLUSIVE.value == "inconclusive"
        assert EvidenceClass.UNSUPPORTED.value == "unsupported"

    def test_claim_evidence_roundtrip(self) -> None:
        ce = ClaimEvidence(chunk_id="c1", relevance_score=0.75, quote="Some text.")
        data = ce.model_dump(mode="json")
        restored = ClaimEvidence.model_validate(data)
        assert restored == ce

    def test_atomic_claim_roundtrip(self) -> None:
        ac = AtomicClaim(
            claim_id="CL-001",
            paper_id="2401.12345",
            source_type="finding",
            statement="The model is fast.",
            evidence_class=EvidenceClass.SUPPORTED,
            evidence=[
                ClaimEvidence(chunk_id="c1", relevance_score=0.8, quote="It is fast.")
            ],
            confidence_score=0.85,
        )
        data = ac.model_dump(mode="json")
        restored = AtomicClaim.model_validate(data)
        assert restored == ac
        assert restored.evidence_class == EvidenceClass.SUPPORTED

    def test_claim_decomposition_roundtrip(self) -> None:
        cd = ClaimDecomposition(
            paper_id="2401.12345",
            title="Test Paper",
            claims=[
                AtomicClaim(
                    claim_id="CL-001",
                    paper_id="2401.12345",
                    source_type="finding",
                    statement="First claim.",
                    evidence_class=EvidenceClass.SUPPORTED,
                    confidence_score=0.9,
                ),
                AtomicClaim(
                    claim_id="CL-002",
                    paper_id="2401.12345",
                    source_type="limitation",
                    statement="Second claim.",
                    evidence_class=EvidenceClass.UNSUPPORTED,
                    confidence_score=0.0,
                ),
            ],
            total_claims=2,
            evidence_summary={"supported": 1, "unsupported": 1},
        )
        data = cd.model_dump(mode="json")
        restored = ClaimDecomposition.model_validate(data)
        assert restored == cd
        assert len(restored.claims) == 2
        assert restored.evidence_summary == {"supported": 1, "unsupported": 1}

    def test_claim_decomposition_json_serializable(self) -> None:
        cd = ClaimDecomposition(
            paper_id="2401.12345",
            title="Test",
            claims=[],
            total_claims=0,
            evidence_summary={},
        )
        json_str = cd.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["paper_id"] == "2401.12345"

    def test_atomic_claim_defaults(self) -> None:
        ac = AtomicClaim(
            claim_id="CL-001",
            paper_id="2401.12345",
            source_type="finding",
            statement="Some claim.",
        )
        assert ac.evidence_class == EvidenceClass.UNSUPPORTED
        assert ac.evidence == []
        assert ac.confidence_score == 0.0


# ── CLI handler tests ────────────────────────────────────────────────


class TestCLIHandler:
    """Tests for the analyze-claims CLI handler."""

    def test_run_analyze_claims_no_summaries(self, tmp_path: Path) -> None:
        """CLI exits with error when no summaries exist."""
        import typer

        from research_pipeline.cli.cmd_analyze_claims import run_analyze_claims

        with (
            patch(
                "research_pipeline.cli.cmd_analyze_claims.load_config"
            ) as mock_config,
            patch("research_pipeline.cli.cmd_analyze_claims.init_run") as mock_init,
            patch(
                "research_pipeline.cli.cmd_analyze_claims.get_stage_dir"
            ) as mock_stage,
        ):
            mock_config.return_value = MagicMock(workspace=str(tmp_path))
            mock_init.return_value = ("test-run", tmp_path / "runs" / "test-run")

            # Create empty summarize dir
            sum_dir = tmp_path / "summarize"
            sum_dir.mkdir(parents=True)
            mock_stage.return_value = sum_dir

            with pytest.raises(typer.Exit):
                run_analyze_claims(workspace=tmp_path, run_id="test-run")

    def test_run_analyze_claims_with_summaries(self, tmp_path: Path) -> None:
        """CLI processes summaries and writes output."""
        from research_pipeline.cli.cmd_analyze_claims import run_analyze_claims

        # Create summary file
        sum_dir = tmp_path / "summarize"
        sum_dir.mkdir(parents=True)
        summary_data = {
            "arxiv_id": "2401.12345",
            "version": "v1",
            "title": "Test Paper",
            "objective": "Test the system performance.",
            "methodology": "We evaluate on standard benchmarks.",
            "findings": ["The system achieves 95% accuracy."],
            "limitations": [],
        }
        summary_path = sum_dir / "2401.12345v1.summary.json"
        summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

        md_dir = tmp_path / "convert"
        md_dir.mkdir(parents=True)

        def mock_get_stage_dir(run_root: Path, stage: str) -> Path:
            if stage == "summarize":
                return sum_dir
            return md_dir

        with (
            patch(
                "research_pipeline.cli.cmd_analyze_claims.load_config"
            ) as mock_config,
            patch("research_pipeline.cli.cmd_analyze_claims.init_run") as mock_init,
            patch(
                "research_pipeline.cli.cmd_analyze_claims.get_stage_dir",
                side_effect=mock_get_stage_dir,
            ),
            patch("research_pipeline.cli.cmd_analyze_claims.write_jsonl") as mock_write,
        ):
            mock_config.return_value = MagicMock(workspace=str(tmp_path))
            mock_init.return_value = ("test-run", tmp_path)

            run_analyze_claims(workspace=tmp_path, run_id="test-run")

            mock_write.assert_called_once()
            written_data = mock_write.call_args[0][1]
            assert len(written_data) == 1
            assert written_data[0]["paper_id"] == "2401.12345"
            assert written_data[0]["total_claims"] > 0
