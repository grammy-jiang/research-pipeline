"""Tests for segment-level memory entries.

Covers the segmentation module, segmented episodic retrieval,
and segmented working memory add.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from research_pipeline.memory.episodic import Episode, EpisodicMemory

# ---------------------------------------------------------------------------
# segmentation module tests
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    """Tests for the token estimation heuristic."""

    def test_empty_string(self) -> None:
        from research_pipeline.memory.segmentation import estimate_tokens

        assert estimate_tokens("") == 1  # +1 minimum

    def test_single_word(self) -> None:
        from research_pipeline.memory.segmentation import estimate_tokens

        result = estimate_tokens("hello")
        assert result >= 1

    def test_rough_approximation(self) -> None:
        from research_pipeline.memory.segmentation import estimate_tokens

        text = " ".join(["word"] * 75)  # 75 words ≈ 100 tokens
        result = estimate_tokens(text)
        assert 90 <= result <= 110


class TestSegmentText:
    """Tests for text segmentation."""

    def test_short_text_no_split(self) -> None:
        from research_pipeline.memory.segmentation import segment_text

        result = segment_text("Short text.", max_tokens=450)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_empty_text(self) -> None:
        from research_pipeline.memory.segmentation import segment_text

        assert segment_text("") == [""]
        assert segment_text("   ") == ["   "]

    def test_long_text_splits(self) -> None:
        from research_pipeline.memory.segmentation import estimate_tokens, segment_text

        # Create text that exceeds 450 tokens (~338 words)
        sentences = [f"Sentence number {i} with some filler words." for i in range(100)]
        long_text = " ".join(sentences)
        assert estimate_tokens(long_text) > 450

        parts = segment_text(long_text, max_tokens=450)
        assert len(parts) > 1
        for part in parts:
            assert estimate_tokens(part) <= 500  # allow small overshoot from overlap

    def test_overlap_provides_continuity(self) -> None:
        from research_pipeline.memory.segmentation import segment_text

        sentences = [f"Sentence {i} with enough words to matter." for i in range(80)]
        long_text = " ".join(sentences)
        parts = segment_text(long_text, max_tokens=100, overlap_sentences=1)
        assert len(parts) >= 2

    def test_no_overlap(self) -> None:
        from research_pipeline.memory.segmentation import segment_text

        sentences = [f"Sentence {i} with enough words to matter." for i in range(80)]
        long_text = " ".join(sentences)
        parts_overlap = segment_text(long_text, max_tokens=100, overlap_sentences=1)
        parts_no_overlap = segment_text(long_text, max_tokens=100, overlap_sentences=0)
        # Without overlap, we should have fewer or equal segments
        assert len(parts_no_overlap) <= len(parts_overlap)

    def test_single_huge_sentence(self) -> None:
        from research_pipeline.memory.segmentation import segment_text

        # One sentence that exceeds limit — should still return it
        huge = "word " * 500
        parts = segment_text(huge, max_tokens=100)
        assert len(parts) >= 1  # at least the original


class TestMemorySegment:
    """Tests for the MemorySegment dataclass."""

    def test_segment_entry_short(self) -> None:
        from research_pipeline.memory.segmentation import segment_entry

        segs = segment_entry("k1", "Short content.", max_tokens=450)
        assert len(segs) == 1
        assert segs[0].parent_key == "k1"
        assert segs[0].segment_index == 0
        assert segs[0].total_segments == 1
        assert segs[0].content == "Short content."

    def test_segment_entry_long(self) -> None:
        from research_pipeline.memory.segmentation import segment_entry

        sentences = [f"Finding {i}: something important." for i in range(100)]
        long = " ".join(sentences)
        segs = segment_entry("big", long, max_tokens=100)
        assert len(segs) > 1
        for i, seg in enumerate(segs):
            assert seg.parent_key == "big"
            assert seg.segment_index == i
            assert seg.total_segments == len(segs)

    def test_segment_entry_preserves_metadata(self) -> None:
        from research_pipeline.memory.segmentation import segment_entry

        segs = segment_entry("k2", "Hello.", metadata={"source": "test"})
        assert segs[0].metadata["source"] == "test"
        assert segs[0].metadata["parent_key"] == "k2"
        assert segs[0].metadata["segment"] == "1/1"


# ---------------------------------------------------------------------------
# EpisodicMemory segmented retrieval
# ---------------------------------------------------------------------------


class TestEpisodicSegmented:
    """Tests for segmented retrieval from EpisodicMemory."""

    def _make_memory(self, tmp_path: Path) -> EpisodicMemory:
        from research_pipeline.memory.episodic import EpisodicMemory

        return EpisodicMemory(db_path=tmp_path / "test_episodic.db")

    def _make_episode(
        self, run_id: str, summary: str = "", decisions: list[str] | None = None
    ) -> Episode:
        from research_pipeline.memory.episodic import Episode

        return Episode(
            run_id=run_id,
            topic="test topic",
            started_at="2026-01-01T00:00:00",
            synthesis_summary=summary,
            key_decisions=decisions or [],
        )

    def test_short_summary_single_segment(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        ep = self._make_episode("r1", summary="Short summary.")
        mem.record_episode(ep)
        segs = mem.get_segmented_summary("r1")
        assert len(segs) == 1
        assert segs[0] == "Short summary."

    def test_long_summary_splits(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        long_summary = " ".join([f"Finding {i}: important result." for i in range(100)])
        ep = self._make_episode("r2", summary=long_summary)
        mem.record_episode(ep)
        segs = mem.get_segmented_summary("r2")
        assert len(segs) > 1

    def test_missing_run_returns_empty(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        assert mem.get_segmented_summary("nonexistent") == []

    def test_empty_summary_returns_empty(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        ep = self._make_episode("r3", summary="")
        mem.record_episode(ep)
        assert mem.get_segmented_summary("r3") == []

    def test_segmented_decisions_short(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        ep = self._make_episode("r4", decisions=["Use BM25.", "Skip SPECTER2."])
        mem.record_episode(ep)
        segs = mem.get_segmented_decisions("r4")
        assert len(segs) == 2
        assert segs[0] == "Use BM25."

    def test_segmented_decisions_long_splits(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        long_decision = " ".join(
            [f"Step {i}: do something important." for i in range(100)]
        )
        ep = self._make_episode("r5", decisions=[long_decision, "Short."])
        mem.record_episode(ep)
        segs = mem.get_segmented_decisions("r5")
        assert len(segs) >= 3  # long splits + "Short."

    def test_segmented_decisions_missing_run(self, tmp_path: Path) -> None:
        mem = self._make_memory(tmp_path)
        assert mem.get_segmented_decisions("nope") == []


# ---------------------------------------------------------------------------
# WorkingMemory segmented add
# ---------------------------------------------------------------------------


class TestWorkingMemorySegmented:
    """Tests for add_segmented in WorkingMemory."""

    def test_short_value_single_item(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=50)
        count = wm.add_segmented("k1", "Short value.", stage="plan")
        assert count == 1
        assert len(wm) == 1
        assert wm.get("k1") is not None

    def test_long_value_splits_into_segments(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=100)
        long_text = " ".join([f"Sentence {i} with filler words." for i in range(100)])
        count = wm.add_segmented("big", long_text, stage="screen", max_tokens=100)
        assert count > 1
        assert len(wm) == count
        # Each segment has seg key like "big__seg0", "big__seg1", ...
        seg0 = wm.get("big__seg0")
        assert seg0 is not None
        assert seg0.metadata.get("parent_key") == "big"

    def test_non_string_value_no_split(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=50)
        count = wm.add_segmented("k2", 42, stage="plan")  # type: ignore[arg-type]
        assert count == 1
        assert wm.get("k2") is not None
        assert wm.get("k2").value == 42  # type: ignore[union-attr]

    def test_segmented_metadata_preserved(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=100)
        long_text = " ".join([f"Finding {i}: result." for i in range(80)])
        count = wm.add_segmented(
            "data",
            long_text,
            stage="extract",
            metadata={"source": "paper1"},
            max_tokens=100,
        )
        assert count > 1
        seg = wm.get("data__seg0")
        assert seg is not None
        assert seg.metadata["source"] == "paper1"
        assert "segment" in seg.metadata

    def test_custom_max_tokens(self) -> None:
        from research_pipeline.memory.working import WorkingMemory

        wm = WorkingMemory(capacity=200)
        # Use sentences so the segmenter has split points
        text = " ".join([f"Sentence {i} with some extra words." for i in range(150)])
        count_small = wm.add_segmented("a", text, max_tokens=50)
        wm.reset()
        count_large = wm.add_segmented("b", text, max_tokens=500)
        assert count_small > count_large
