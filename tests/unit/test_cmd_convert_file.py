"""Tests for the standalone convert-file CLI command."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from research_pipeline.cli.cmd_convert_file import run_convert_file
from research_pipeline.models.conversion import ConvertManifestEntry


def _make_entry(
    status: str = "converted",
    error: str | None = None,
) -> ConvertManifestEntry:
    """Build a ConvertManifestEntry for testing."""
    return ConvertManifestEntry(
        arxiv_id="2401.00001",
        version="v1",
        pdf_path="/tmp/test.pdf",
        pdf_sha256="abc123",
        markdown_path="/tmp/test.md",
        converter_name="docling",
        converter_version="2.0.0",
        converter_config_hash="deadbeef",
        converted_at=datetime.now(tz=UTC),
        status=status,
        error=error,
    )


class TestFileNotFound:
    """Tests for missing PDF file."""

    def test_exits_with_code_1(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.pdf"
        with pytest.raises(typer.Exit) as exc_info:
            run_convert_file(missing)
        assert exc_info.value.exit_code == 1

    def test_prints_error_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "missing.pdf"
        with pytest.raises(typer.Exit):
            run_convert_file(missing)
        assert "PDF file not found" in capsys.readouterr().err


_BACKEND = "research_pipeline.conversion.docling_backend.DoclingBackend"


class TestNonPdfExtension:
    """Tests for files without .pdf extension."""

    @patch(_BACKEND)
    def test_warns_but_continues(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        txt_file = tmp_path / "paper.txt"
        txt_file.write_text("not a pdf")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry()
        mock_backend_cls.return_value = mock_backend

        run_convert_file(txt_file)
        assert "does not have .pdf extension" in capsys.readouterr().err


class TestSuccessfulConversion:
    """Tests for successful PDF → Markdown conversion."""

    @patch(_BACKEND)
    def test_converted_status(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry(status="converted")
        mock_backend_cls.return_value = mock_backend

        run_convert_file(pdf)

        out = capsys.readouterr().out
        assert "OK:" in out
        assert "paper.md" in out

    @patch(_BACKEND)
    def test_skipped_exists_status(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry(status="skipped_exists")
        mock_backend_cls.return_value = mock_backend

        run_convert_file(pdf)

        out = capsys.readouterr().out
        assert "OK:" in out


class TestCustomOutputDir:
    """Tests for specifying a custom output directory."""

    @patch(_BACKEND)
    def test_output_uses_custom_dir(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out_dir = tmp_path / "output"

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry()
        mock_backend_cls.return_value = mock_backend

        run_convert_file(pdf, output_dir=out_dir)

        # Verify backend called with custom output dir
        mock_backend.convert.assert_called_once_with(pdf.resolve(), out_dir.resolve())
        assert out_dir.exists()

    @patch(_BACKEND)
    def test_default_output_is_pdf_parent(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry()
        mock_backend_cls.return_value = mock_backend

        run_convert_file(pdf)

        call_args = mock_backend.convert.call_args
        assert call_args[0][1] == pdf.resolve().parent


class TestDoclingImportError:
    """Tests for missing Docling dependency."""

    @patch(_BACKEND)
    def test_exits_with_code_1(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.side_effect = ImportError("No module named 'docling'")
        mock_backend_cls.return_value = mock_backend

        with pytest.raises(typer.Exit) as exc_info:
            run_convert_file(pdf)
        assert exc_info.value.exit_code == 1

    @patch(_BACKEND)
    def test_prints_install_hint(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.side_effect = ImportError("No module named 'docling'")
        mock_backend_cls.return_value = mock_backend

        with pytest.raises(typer.Exit):
            run_convert_file(pdf)
        assert "pipx inject" in capsys.readouterr().err


class TestFailedConversion:
    """Tests for conversion failures."""

    @patch(_BACKEND)
    def test_exits_with_code_1(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry(
            status="failed", error="Parse error"
        )
        mock_backend_cls.return_value = mock_backend

        with pytest.raises(typer.Exit) as exc_info:
            run_convert_file(pdf)
        assert exc_info.value.exit_code == 1

    @patch(_BACKEND)
    def test_prints_error_detail(
        self,
        mock_backend_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        mock_backend = MagicMock()
        mock_backend.convert.return_value = _make_entry(
            status="failed", error="Parse error"
        )
        mock_backend_cls.return_value = mock_backend

        with pytest.raises(typer.Exit):
            run_convert_file(pdf)
        err = capsys.readouterr().err
        assert "Failed" in err
        assert "Parse error" in err
