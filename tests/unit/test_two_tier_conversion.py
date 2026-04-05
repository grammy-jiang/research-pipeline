"""Unit tests for two-tier conversion (Phase 7)."""

from research_pipeline.models.conversion import ConvertManifestEntry


class TestConvertManifestTier:
    """Tests for the tier field on ConvertManifestEntry."""

    def test_default_tier_is_rough(self) -> None:
        entry = ConvertManifestEntry(
            arxiv_id="2401.12345",
            version="v1",
            pdf_path="/tmp/test.pdf",
            pdf_sha256="abc123",
            markdown_path="/tmp/test.md",
            converter_name="pymupdf4llm",
            converter_version="1.0.0",
            converter_config_hash="hash",
            converted_at="2024-01-01T00:00:00Z",
            status="converted",
        )
        assert entry.tier == "rough"

    def test_fine_tier(self) -> None:
        entry = ConvertManifestEntry(
            arxiv_id="2401.12345",
            version="v1",
            pdf_path="/tmp/test.pdf",
            pdf_sha256="abc123",
            markdown_path="/tmp/test.md",
            converter_name="docling",
            converter_version="2.0.0",
            converter_config_hash="hash",
            converted_at="2024-01-01T00:00:00Z",
            status="converted",
            tier="fine",
        )
        assert entry.tier == "fine"

    def test_backward_compatible_roundtrip(self) -> None:
        """Old manifests without tier field should deserialize correctly."""
        data = {
            "arxiv_id": "2401.12345",
            "version": "v1",
            "pdf_path": "/tmp/test.pdf",
            "pdf_sha256": "abc123",
            "markdown_path": "/tmp/test.md",
            "converter_name": "pymupdf4llm",
            "converter_version": "1.0.0",
            "converter_config_hash": "hash",
            "converted_at": "2024-01-01T00:00:00Z",
            "status": "converted",
        }
        entry = ConvertManifestEntry(**data)
        assert entry.tier == "rough"

    def test_serialization_includes_tier(self) -> None:
        entry = ConvertManifestEntry(
            arxiv_id="2401.12345",
            version="v1",
            pdf_path="/tmp/test.pdf",
            pdf_sha256="abc123",
            markdown_path="/tmp/test.md",
            converter_name="marker",
            converter_version="1.0.0",
            converter_config_hash="hash",
            converted_at="2024-01-01T00:00:00Z",
            status="converted",
            tier="fine",
        )
        data = entry.model_dump(mode="json")
        assert data["tier"] == "fine"
