"""Tests for the analyze command (cmd_analyze)."""

import json
from pathlib import Path

import pytest

from research_pipeline.cli.cmd_analyze import (
    RATING_DIMENSIONS,
    _discover_papers,
    _generate_prompt,
    _load_research_topic,
    _validate_analysis_json,
)


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Create a mock run directory structure."""
    # Plan stage
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "query_plan.json").write_text(
        json.dumps({"topic": "memory systems for AI agents"})
    )

    # Convert stage (markdown files)
    convert_dir = tmp_path / "convert" / "markdown"
    convert_dir.mkdir(parents=True)
    (convert_dir / "2401.12345.md").write_text("# Paper about memory systems")
    (convert_dir / "2401.67890.md").write_text("# Paper about AI agents")

    # Analysis stage dir
    (tmp_path / "analysis").mkdir()

    return tmp_path


class TestDiscoverPapers:
    def test_finds_papers_in_convert_dir(self, run_root: Path) -> None:
        papers = _discover_papers(run_root)
        assert len(papers) == 2
        ids = {p["arxiv_id"] for p in papers}
        assert ids == {"2401.12345", "2401.67890"}

    def test_empty_when_no_convert_dir(self, tmp_path: Path) -> None:
        papers = _discover_papers(tmp_path)
        assert papers == []

    def test_finds_papers_in_convert_rough(self, tmp_path: Path) -> None:
        rough_dir = tmp_path / "convert_rough"
        rough_dir.mkdir()
        (rough_dir / "2401.99999.md").write_text("# rough converted")
        papers = _discover_papers(tmp_path)
        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2401.99999"


class TestLoadResearchTopic:
    def test_loads_topic(self, run_root: Path) -> None:
        topic = _load_research_topic(run_root)
        assert topic == "memory systems for AI agents"

    def test_empty_when_no_plan(self, tmp_path: Path) -> None:
        (tmp_path / "plan").mkdir()
        topic = _load_research_topic(tmp_path)
        assert topic == ""


class TestGeneratePrompt:
    def test_prompt_structure(self, run_root: Path) -> None:
        paper = {"arxiv_id": "2401.12345", "path": "/tmp/paper.md"}
        prompt = _generate_prompt(paper, "AI memory", run_root)
        assert prompt["arxiv_id"] == "2401.12345"
        assert prompt["research_topic"] == "AI memory"
        assert "paper_path" in prompt
        assert "prompt" in prompt
        assert "output_json" in prompt
        assert "output_markdown" in prompt


class TestValidateAnalysisJson:
    def _make_valid_analysis(self) -> dict:
        """Create a valid analysis JSON."""
        return {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "ratings": {
                dim: {
                    "score": 4,
                    "justification": ("This is a well written detailed justification"),
                }
                for dim in RATING_DIMENSIONS
            },
            "methodology_assessment": "Strong methodology.",
            "key_findings": [
                {"finding": "Finding 1", "confidence": "high"},
                {"finding": "Finding 2", "confidence": "medium"},
            ],
            "strengths": ["Good approach"],
            "weaknesses": ["Limited dataset"],
            "limitations": ["Single domain"],
            "evidence_quotes": ["quote 1", "quote 2"],
            "key_contributions": ["Contribution 1"],
            "reproducibility": {"code_available": True},
            "relevance_to_topic": "Highly relevant",
        }

    def test_valid_analysis(self, tmp_path: Path) -> None:
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(self._make_valid_analysis()))
        errors = _validate_analysis_json(path)
        assert errors == []

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        data = {"arxiv_id": "2401.12345", "title": "Test"}
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(data))
        errors = _validate_analysis_json(path)
        assert any("Missing required fields" in e for e in errors)

    def test_invalid_rating_score(self, tmp_path: Path) -> None:
        data = self._make_valid_analysis()
        data["ratings"]["methodology"]["score"] = 10
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(data))
        errors = _validate_analysis_json(path)
        assert any("must be int 1-5" in e for e in errors)

    def test_short_justification(self, tmp_path: Path) -> None:
        data = self._make_valid_analysis()
        data["ratings"]["methodology"]["justification"] = "ok"
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(data))
        errors = _validate_analysis_json(path)
        assert any("justification too short" in e for e in errors)

    def test_invalid_confidence(self, tmp_path: Path) -> None:
        data = self._make_valid_analysis()
        data["key_findings"] = [{"finding": "F1", "confidence": "maybe"}]
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(data))
        errors = _validate_analysis_json(path)
        assert any("confidence must be high/medium/low" in e for e in errors)

    def test_empty_evidence_quotes(self, tmp_path: Path) -> None:
        data = self._make_valid_analysis()
        data["evidence_quotes"] = []
        path = tmp_path / "test_analysis.json"
        path.write_text(json.dumps(data))
        errors = _validate_analysis_json(path)
        assert any("evidence_quotes should not be empty" in e for e in errors)

    def test_unreadable_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        errors = _validate_analysis_json(path)
        assert len(errors) == 1
        assert "Cannot read" in errors[0]

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        errors = _validate_analysis_json(path)
        assert len(errors) == 1
        assert "Cannot read" in errors[0]
