"""Unit tests for the MinerU (magic-pdf) PDF-to-Markdown conversion backend.

Tests registration, fingerprint format, convert paths (success, skip, force,
ImportError, generic error with CLI fallback). All magic_pdf imports are mocked
since the package is not installed in the test environment.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from research_pipeline.conversion.registry import (
    _ensure_builtins_registered,
    get_backend,
    list_backends,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """Create a minimal fake PDF for testing."""
    pdf = tmp_path / "2401.00001v1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content for testing")
    return pdf


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestMinerURegistration:
    """Verify MinerU is registered in the backend registry."""

    def test_registered_after_ensure(self) -> None:
        _ensure_builtins_registered()
        assert "mineru" in list_backends()

    def test_get_backend_returns_instance(self) -> None:
        _ensure_builtins_registered()
        backend = get_backend("mineru")
        assert backend.__class__.__name__ == "MinerUBackend"

    def test_get_backend_with_kwargs(self) -> None:
        _ensure_builtins_registered()
        backend = get_backend("mineru", parse_method="ocr", timeout_seconds=120)
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        assert isinstance(backend, MinerUBackend)
        assert backend.parse_method == "ocr"
        assert backend.timeout_seconds == 120


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestMinerUFingerprint:
    """Verify fingerprint format and variation."""

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()
        fp = b.fingerprint()
        parts = fp.split("/")
        assert len(parts) == 3
        assert parts[0] == "mineru"

    def test_fingerprint_includes_version(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()
        b._version = "0.10.0"
        fp = b.fingerprint()
        assert fp.startswith("mineru/0.10.0/")

    def test_fingerprint_varies_by_parse_method(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b1 = MinerUBackend(parse_method="auto")
        b2 = MinerUBackend(parse_method="ocr")
        assert b1.fingerprint() != b2.fingerprint()

    def test_fingerprint_varies_by_timeout(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b1 = MinerUBackend(timeout_seconds=300)
        b2 = MinerUBackend(timeout_seconds=600)
        assert b1.fingerprint() != b2.fingerprint()

    def test_version_fallback_unknown(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()
        # magic-pdf is not installed, so version should fall back to "unknown"
        assert b.version == "unknown"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestMinerUConstructor:
    """Verify constructor defaults and overrides."""

    def test_defaults(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()
        assert b.parse_method == "auto"
        assert b.timeout_seconds == 600

    def test_custom_parse_method(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        for method in ("auto", "ocr", "txt"):
            b = MinerUBackend(parse_method=method)
            assert b.parse_method == method

    def test_custom_timeout(self) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend(timeout_seconds=120)
        assert b.timeout_seconds == 120


# ---------------------------------------------------------------------------
# Convert: skip exists
# ---------------------------------------------------------------------------


class TestMinerUSkipExists:
    """Verify skip-when-already-exists behaviour."""

    def test_skip_existing(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = MinerUBackend()
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "mineru"
        assert entry.arxiv_id == "2401.00001"
        assert entry.version == "v1"


# ---------------------------------------------------------------------------
# Convert: force mode
# ---------------------------------------------------------------------------


class TestMinerUForceMode:
    """Verify force mode removes existing output."""

    @patch(
        "research_pipeline.conversion.mineru_backend.MinerUBackend._convert_python_api"
    )
    def test_force_removes_existing(
        self, mock_api: MagicMock, pdf_file: Path, output_dir: Path
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Old content", encoding="utf-8")

        mock_api.return_value = "# New content from MinerU"

        b = MinerUBackend()
        entry = b.convert(pdf_file, output_dir, force=True)
        assert entry.status == "converted"
        assert md_path.read_text(encoding="utf-8") == "# New content from MinerU"


# ---------------------------------------------------------------------------
# Convert: successful Python API
# ---------------------------------------------------------------------------


class TestMinerUConvertSuccess:
    """Verify successful conversion via mocked Python API."""

    @patch(
        "research_pipeline.conversion.mineru_backend.MinerUBackend._convert_python_api"
    )
    def test_convert_success(
        self, mock_api: MagicMock, pdf_file: Path, output_dir: Path
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        mock_api.return_value = "# Converted markdown content"

        b = MinerUBackend()
        entry = b.convert(pdf_file, output_dir)

        assert entry.status == "converted"
        assert entry.converter_name == "mineru"
        assert entry.arxiv_id == "2401.00001"
        assert entry.version == "v1"
        assert entry.error is None

        md_path = output_dir / "2401.00001v1.md"
        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8") == "# Converted markdown content"

    @patch(
        "research_pipeline.conversion.mineru_backend.MinerUBackend._convert_python_api"
    )
    def test_convert_no_version_suffix(
        self, mock_api: MagicMock, tmp_path: Path
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        pdf = tmp_path / "2401.00001.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "out"
        out.mkdir()

        mock_api.return_value = "# Content"
        b = MinerUBackend()
        entry = b.convert(pdf, out)

        assert entry.status == "converted"
        # No version suffix → defaults
        assert entry.arxiv_id == "2401.00001"
        assert entry.version == "v1"


# ---------------------------------------------------------------------------
# Convert: ImportError (magic-pdf not installed)
# ---------------------------------------------------------------------------


class TestMinerUImportError:
    """Verify graceful handling when magic-pdf is not installed."""

    def test_import_error_returns_failed(
        self, pdf_file: Path, output_dir: Path
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()

        # _convert_python_api raises ImportError when magic_pdf is not installed
        with patch.object(
            b,
            "_convert_python_api",
            side_effect=ImportError("No module named 'magic_pdf'"),
        ):
            entry = b.convert(pdf_file, output_dir)

        assert entry.status == "failed"
        assert entry.error is not None
        assert "magic-pdf is not installed" in entry.error
        assert len(entry.warnings) == 1


# ---------------------------------------------------------------------------
# Convert: generic exception → CLI fallback
# ---------------------------------------------------------------------------


class TestMinerUCliFallback:
    """Verify CLI fallback when Python API fails with non-ImportError."""

    @patch("research_pipeline.conversion.mineru_backend.subprocess.run")
    def test_cli_fallback_success(
        self,
        mock_run: MagicMock,
        pdf_file: Path,
        output_dir: Path,
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()

        # Make Python API fail with a generic error
        with patch.object(
            b, "_convert_python_api", side_effect=RuntimeError("API broken")
        ):
            # CLI fallback should succeed
            def fake_cli_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                # Create output dir structure as magic-pdf CLI would
                cli_out = output_dir / (pdf_file.stem + "_cli_out")
                cli_out.mkdir(parents=True, exist_ok=True)
                sub = cli_out / pdf_file.stem / "auto"
                sub.mkdir(parents=True, exist_ok=True)
                md = sub / "content.md"
                md.write_text("# CLI converted content", encoding="utf-8")
                return MagicMock(returncode=0, stderr="", stdout="")

            mock_run.side_effect = fake_cli_run
            entry = b.convert(pdf_file, output_dir)

        assert entry.status == "converted"
        assert "CLI fallback" in entry.warnings[0]

    @patch("research_pipeline.conversion.mineru_backend.subprocess.run")
    def test_cli_fallback_failure(
        self,
        mock_run: MagicMock,
        pdf_file: Path,
        output_dir: Path,
    ) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend()

        with patch.object(
            b, "_convert_python_api", side_effect=RuntimeError("API broken")
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stderr="CLI error occurred", stdout=""
            )
            entry = b.convert(pdf_file, output_dir)

        assert entry.status == "failed"
        assert entry.error is not None
        assert "MinerU conversion failed" in entry.error

    @patch("research_pipeline.conversion.mineru_backend.subprocess.run")
    def test_cli_fallback_timeout(
        self,
        mock_run: MagicMock,
        pdf_file: Path,
        output_dir: Path,
    ) -> None:
        import subprocess

        from research_pipeline.conversion.mineru_backend import MinerUBackend

        b = MinerUBackend(timeout_seconds=10)

        with patch.object(
            b, "_convert_python_api", side_effect=RuntimeError("API broken")
        ):
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="magic-pdf", timeout=10
            )
            entry = b.convert(pdf_file, output_dir)

        assert entry.status == "failed"
        assert entry.error is not None


# ---------------------------------------------------------------------------
# ArXiv ID parsing from filename
# ---------------------------------------------------------------------------


class TestMinerUArxivIdParsing:
    """Verify arxiv_id and version parsing from PDF filenames."""

    @patch(
        "research_pipeline.conversion.mineru_backend.MinerUBackend._convert_python_api"
    )
    def test_versioned_filename(self, mock_api: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        pdf = tmp_path / "2401.12345v2.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "out"
        out.mkdir()

        mock_api.return_value = "# Content"
        b = MinerUBackend()
        entry = b.convert(pdf, out)

        assert entry.arxiv_id == "2401.12345"
        assert entry.version == "v2"

    @patch(
        "research_pipeline.conversion.mineru_backend.MinerUBackend._convert_python_api"
    )
    def test_unversioned_filename(self, mock_api: MagicMock, tmp_path: Path) -> None:
        from research_pipeline.conversion.mineru_backend import MinerUBackend

        pdf = tmp_path / "2401.12345.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = tmp_path / "out"
        out.mkdir()

        mock_api.return_value = "# Content"
        b = MinerUBackend()
        entry = b.convert(pdf, out)

        assert entry.arxiv_id == "2401.12345"
        assert entry.version == "v1"
