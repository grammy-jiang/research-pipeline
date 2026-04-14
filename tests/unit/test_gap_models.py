"""Unit tests for GapRecord and EvidenceMap models."""

from research_pipeline.models.gap import EvidenceMap, EvidenceMapEntry, GapRecord


class TestGapRecord:
    """Tests for the GapRecord model."""

    def test_roundtrip_academic(self) -> None:
        """Academic gap round-trips through JSON serialization."""
        gap = GapRecord(
            gap_id="GAP-001",
            gap_type="ACADEMIC",
            severity="HIGH",
            description="No papers cover multi-hop retrieval",
            impact="Core design question unanswered",
            related_papers=["2401.12345"],
            suggested_queries=["multi-hop retrieval augmented generation"],
        )
        data = gap.model_dump(mode="json")
        restored = GapRecord.model_validate(data)
        assert restored == gap

    def test_roundtrip_engineering(self) -> None:
        """Engineering gap round-trips through JSON serialization."""
        gap = GapRecord(
            gap_id="GAP-002",
            gap_type="ENGINEERING",
            severity="MEDIUM",
            description="No benchmark for cache invalidation",
            resolution="Implement custom benchmark suite",
        )
        data = gap.model_dump(mode="json")
        restored = GapRecord.model_validate(data)
        assert restored == gap

    def test_defaults(self) -> None:
        """Default values are applied correctly."""
        gap = GapRecord(
            gap_id="GAP-003",
            gap_type="ACADEMIC",
            severity="LOW",
            description="Limited coverage of edge cases",
        )
        assert gap.impact == ""
        assert gap.related_papers == []
        assert gap.suggested_queries == []
        assert gap.resolution == ""
        assert gap.resolved is False
        assert gap.resolved_in_iteration is None

    def test_resolved_fields(self) -> None:
        """Resolved gap has iteration number."""
        gap = GapRecord(
            gap_id="GAP-004",
            gap_type="ACADEMIC",
            severity="HIGH",
            description="Gap resolved in iteration 2",
            resolved=True,
            resolved_in_iteration=2,
        )
        assert gap.resolved is True
        assert gap.resolved_in_iteration == 2


class TestEvidenceMapEntry:
    """Tests for the EvidenceMapEntry model."""

    def test_roundtrip(self) -> None:
        """Entry round-trips through JSON."""
        entry = EvidenceMapEntry(
            paper_id="2401.12345",
            aspect="memory retrieval",
            covered=True,
            section_ref="§3.2",
            confidence="HIGH",
        )
        data = entry.model_dump(mode="json")
        restored = EvidenceMapEntry.model_validate(data)
        assert restored == entry

    def test_defaults(self) -> None:
        """Default values applied."""
        entry = EvidenceMapEntry(
            paper_id="2401.99999",
            aspect="indexing",
        )
        assert entry.covered is False
        assert entry.section_ref == ""
        assert entry.confidence == "MEDIUM"


class TestEvidenceMap:
    """Tests for the EvidenceMap model."""

    def _make_map(self) -> EvidenceMap:
        """Create a sample evidence map for testing."""
        return EvidenceMap(
            research_question="How to build AI memory systems?",
            aspects=["retrieval", "storage", "forgetting"],
            papers=["2401.11111", "2401.22222"],
            entries=[
                EvidenceMapEntry(
                    paper_id="2401.11111",
                    aspect="retrieval",
                    covered=True,
                    section_ref="§3",
                    confidence="HIGH",
                ),
                EvidenceMapEntry(
                    paper_id="2401.22222",
                    aspect="storage",
                    covered=True,
                    section_ref="§4",
                    confidence="MEDIUM",
                ),
                EvidenceMapEntry(
                    paper_id="2401.11111",
                    aspect="storage",
                    covered=False,
                    confidence="LOW",
                ),
            ],
        )

    def test_roundtrip(self) -> None:
        """Map round-trips through JSON."""
        emap = self._make_map()
        data = emap.model_dump(mode="json")
        restored = EvidenceMap.model_validate(data)
        assert restored == emap

    def test_coverage_for_aspect(self) -> None:
        """Filter entries by aspect."""
        emap = self._make_map()
        storage = emap.coverage_for_aspect("storage")
        assert len(storage) == 2
        assert all(e.aspect == "storage" for e in storage)

    def test_coverage_for_paper(self) -> None:
        """Filter entries by paper."""
        emap = self._make_map()
        paper1 = emap.coverage_for_paper("2401.11111")
        assert len(paper1) == 2
        assert all(e.paper_id == "2401.11111" for e in paper1)

    def test_uncovered_aspects(self) -> None:
        """Identify aspects with no coverage."""
        emap = self._make_map()
        uncovered = emap.uncovered_aspects()
        assert uncovered == ["forgetting"]

    def test_all_covered(self) -> None:
        """No uncovered aspects when everything is covered."""
        emap = EvidenceMap(
            research_question="Test",
            aspects=["a", "b"],
            papers=["p1"],
            entries=[
                EvidenceMapEntry(paper_id="p1", aspect="a", covered=True),
                EvidenceMapEntry(paper_id="p1", aspect="b", covered=True),
            ],
        )
        assert emap.uncovered_aspects() == []

    def test_empty_map(self) -> None:
        """Empty map has all aspects uncovered."""
        emap = EvidenceMap(
            research_question="Test",
            aspects=["x", "y"],
            papers=[],
            entries=[],
        )
        assert emap.uncovered_aspects() == ["x", "y"]
