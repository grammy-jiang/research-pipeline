"""Unit tests for storage.global_index (Phase 8)."""

from pathlib import Path

import pytest

from research_pipeline.storage.global_index import GlobalPaperIndex


@pytest.fixture
def index(tmp_path: Path) -> GlobalPaperIndex:
    """Create a temporary global index."""
    db_path = tmp_path / "test_index.db"
    idx = GlobalPaperIndex(db_path=db_path)
    yield idx
    idx.close()


class TestGlobalPaperIndex:
    """Tests for GlobalPaperIndex."""

    def test_register_and_is_known(self, index: GlobalPaperIndex) -> None:
        assert not index.is_known("2401.12345")
        index.register_paper("2401.12345", run_id="run1", stage="search")
        assert index.is_known("2401.12345")

    def test_find_known_ids(self, index: GlobalPaperIndex) -> None:
        index.register_paper("2401.11111", run_id="run1", stage="search")
        index.register_paper("2401.22222", run_id="run1", stage="search")
        found = index.find_known_ids(["2401.11111", "2401.33333"])
        assert found == {"2401.11111"}

    def test_find_known_ids_empty(self, index: GlobalPaperIndex) -> None:
        assert index.find_known_ids([]) == set()

    def test_find_artifact(self, index: GlobalPaperIndex) -> None:
        index.register_paper(
            "2401.12345",
            run_id="run1",
            stage="download",
            pdf_path="/tmp/test.pdf",
            pdf_sha256="abc123",
        )
        artifact = index.find_artifact("2401.12345", "download")
        assert artifact is not None
        assert artifact["pdf_path"] == "/tmp/test.pdf"
        assert artifact["pdf_sha256"] == "abc123"

    def test_find_artifact_not_found(self, index: GlobalPaperIndex) -> None:
        assert index.find_artifact("2401.99999", "download") is None

    def test_list_papers(self, index: GlobalPaperIndex) -> None:
        index.register_paper(
            "2401.11111", run_id="run1", stage="search", title="Paper A"
        )
        index.register_paper(
            "2401.22222", run_id="run1", stage="search", title="Paper B"
        )
        papers = index.list_papers()
        assert len(papers) == 2

    def test_list_papers_empty(self, index: GlobalPaperIndex) -> None:
        papers = index.list_papers()
        assert papers == []

    def test_garbage_collect_removes_stale(
        self, index: GlobalPaperIndex, tmp_path: Path
    ) -> None:
        # Register with a non-existent path
        index.register_paper(
            "2401.12345",
            run_id="run1",
            stage="download",
            pdf_path="/nonexistent/path.pdf",
        )
        removed = index.garbage_collect()
        assert removed == 1
        assert not index.is_known("2401.12345")

    def test_garbage_collect_keeps_existing(
        self, index: GlobalPaperIndex, tmp_path: Path
    ) -> None:
        # Create an actual file
        pdf = tmp_path / "real.pdf"
        pdf.write_text("fake pdf")
        index.register_paper(
            "2401.12345",
            run_id="run1",
            stage="download",
            pdf_path=str(pdf),
        )
        removed = index.garbage_collect()
        assert removed == 0
        assert index.is_known("2401.12345")

    def test_garbage_collect_keeps_no_paths(self, index: GlobalPaperIndex) -> None:
        # No paths at all → not stale
        index.register_paper("2401.12345", run_id="run1", stage="search")
        removed = index.garbage_collect()
        assert removed == 0

    def test_register_updates_existing(self, index: GlobalPaperIndex) -> None:
        index.register_paper("2401.12345", run_id="run1", stage="search")
        index.register_paper(
            "2401.12345",
            run_id="run1",
            stage="download",
            pdf_path="/tmp/new.pdf",
        )
        artifact = index.find_artifact("2401.12345", "download")
        assert artifact is not None
        assert artifact["pdf_path"] == "/tmp/new.pdf"

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db_path = tmp_path / "reopen.db"
        idx1 = GlobalPaperIndex(db_path=db_path)
        idx1.register_paper("2401.12345", run_id="run1", stage="search")
        idx1.close()

        idx2 = GlobalPaperIndex(db_path=db_path)
        assert idx2.is_known("2401.12345")
        idx2.close()
