"""Unit tests for the 5 online PDF-to-Markdown conversion backends.

Tests validation, fingerprint format, registry presence, skip-exists logic,
and convert-error paths. Actual API calls are mocked.
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


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Create a minimal fake PDF for testing."""
    pdf = tmp_path / "2401.00001v1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content for testing")
    return pdf


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# MathpixBackend
# ---------------------------------------------------------------------------


class TestMathpixBackend:
    """Tests for the Mathpix cloud backend."""

    def test_missing_credentials_raises(self) -> None:
        from research_pipeline.conversion.mathpix_backend import MathpixBackend

        with pytest.raises(ValueError, match="app_id.*app_key"):
            MathpixBackend()

    def test_missing_app_key_raises(self) -> None:
        from research_pipeline.conversion.mathpix_backend import MathpixBackend

        with pytest.raises(ValueError, match="app_id.*app_key"):
            MathpixBackend(app_id="id")

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.mathpix_backend import MathpixBackend

        b = MathpixBackend(app_id="test_id", app_key="test_key")
        fp = b.fingerprint()
        assert fp.startswith("mathpix/cloud/")
        assert len(fp.split("/")) == 3

    def test_registered_in_registry(self) -> None:
        _ensure_builtins_registered()
        assert "mathpix" in list_backends()

    def test_skip_exists(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.mathpix_backend import MathpixBackend

        # Pre-create output file
        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = MathpixBackend(app_id="test_id", app_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "mathpix"

    @patch("research_pipeline.conversion.mathpix_backend.requests")
    def test_convert_api_error(
        self, mock_requests: MagicMock, pdf_file: Path, output_dir: Path
    ) -> None:
        from research_pipeline.conversion.mathpix_backend import MathpixBackend

        mock_requests.post.side_effect = RuntimeError("API unavailable")
        b = MathpixBackend(app_id="test_id", app_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "failed"
        assert entry.error is not None
        assert "API unavailable" in entry.error


# ---------------------------------------------------------------------------
# DatalabBackend
# ---------------------------------------------------------------------------


class TestDatalabBackend:
    """Tests for the Datalab (hosted Marker) cloud backend."""

    def test_missing_api_key_raises(self) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        with pytest.raises(ValueError, match="api_key"):
            DatalabBackend()

    def test_invalid_mode_raises(self) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        with pytest.raises(ValueError, match="Invalid Datalab mode"):
            DatalabBackend(api_key="test_key", mode="invalid")

    def test_valid_modes(self) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        for mode in ("fast", "balanced", "accurate"):
            b = DatalabBackend(api_key="test_key", mode=mode)
            assert b.mode == mode

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        b = DatalabBackend(api_key="test_key")
        fp = b.fingerprint()
        assert fp.startswith("datalab/cloud/")
        assert len(fp.split("/")) == 3

    def test_registered_in_registry(self) -> None:
        _ensure_builtins_registered()
        assert "datalab" in list_backends()

    def test_skip_exists(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = DatalabBackend(api_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "datalab"

    def test_convert_import_error(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.datalab_backend import DatalabBackend

        b = DatalabBackend(api_key="test_key")
        # datalab_sdk is not installed in test env, so convert should fail gracefully
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "failed"
        assert entry.error is not None


# ---------------------------------------------------------------------------
# LlamaParseBackend
# ---------------------------------------------------------------------------


class TestLlamaParseBackend:
    """Tests for the LlamaParse cloud backend."""

    def test_missing_api_key_raises(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        with pytest.raises(ValueError, match="api_key"):
            LlamaParseBackend()

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b = LlamaParseBackend(api_key="test_key_12345")
        fp = b.fingerprint()
        assert fp.startswith("llamaparse/cloud/")
        assert len(fp.split("/")) == 3

    def test_fingerprint_does_not_contain_api_key(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b = LlamaParseBackend(api_key="secret_api_key_value")
        fp = b.fingerprint()
        assert "secret" not in fp
        assert "api_key" not in fp

    def test_registered_in_registry(self) -> None:
        _ensure_builtins_registered()
        assert "llamaparse" in list_backends()

    def test_skip_exists(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = LlamaParseBackend(api_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "llamaparse"

    def test_convert_import_error(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b = LlamaParseBackend(api_key="test_key")
        # llama_cloud is not installed in test env
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "failed"
        assert entry.error is not None

    def test_tier_default(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b = LlamaParseBackend(api_key="test_key")
        assert b.tier == "agentic"

    def test_tier_custom(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b = LlamaParseBackend(api_key="test_key", tier="cost-effective")
        assert b.tier == "cost-effective"

    def test_fingerprint_varies_by_tier(self) -> None:
        from research_pipeline.conversion.llamaparse_backend import LlamaParseBackend

        b1 = LlamaParseBackend(api_key="test_key", tier="fast")
        b2 = LlamaParseBackend(api_key="test_key", tier="agentic")
        assert b1.fingerprint() != b2.fingerprint()


# ---------------------------------------------------------------------------
# MistralOcrBackend
# ---------------------------------------------------------------------------


class TestMistralOcrBackend:
    """Tests for the Mistral OCR cloud backend."""

    def test_missing_api_key_raises(self) -> None:
        from research_pipeline.conversion.mistral_ocr_backend import MistralOcrBackend

        with pytest.raises(ValueError, match="api_key"):
            MistralOcrBackend()

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.mistral_ocr_backend import MistralOcrBackend

        b = MistralOcrBackend(api_key="test_key")
        fp = b.fingerprint()
        assert fp.startswith("mistral_ocr/cloud/")
        assert len(fp.split("/")) == 3

    def test_custom_model(self) -> None:
        from research_pipeline.conversion.mistral_ocr_backend import MistralOcrBackend

        b = MistralOcrBackend(api_key="test_key", model="custom-model")
        assert b.model == "custom-model"

    def test_registered_in_registry(self) -> None:
        _ensure_builtins_registered()
        assert "mistral_ocr" in list_backends()

    def test_skip_exists(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.mistral_ocr_backend import MistralOcrBackend

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = MistralOcrBackend(api_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "mistral_ocr"

    def test_convert_import_error(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.mistral_ocr_backend import MistralOcrBackend

        b = MistralOcrBackend(api_key="test_key")
        # mistralai is not installed in test env
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "failed"
        assert entry.error is not None


# ---------------------------------------------------------------------------
# OpenAIVisionBackend
# ---------------------------------------------------------------------------


class TestOpenAIVisionBackend:
    """Tests for the OpenAI GPT-4o vision backend."""

    def test_missing_api_key_raises(self) -> None:
        from research_pipeline.conversion.openai_vision_backend import (
            OpenAIVisionBackend,
        )

        with pytest.raises(ValueError, match="api_key"):
            OpenAIVisionBackend()

    def test_fingerprint_format(self) -> None:
        from research_pipeline.conversion.openai_vision_backend import (
            OpenAIVisionBackend,
        )

        b = OpenAIVisionBackend(api_key="test_key")
        fp = b.fingerprint()
        assert fp.startswith("openai_vision/cloud/")
        assert len(fp.split("/")) == 3

    def test_custom_model(self) -> None:
        from research_pipeline.conversion.openai_vision_backend import (
            OpenAIVisionBackend,
        )

        b = OpenAIVisionBackend(api_key="test_key", model="gpt-4o-mini")
        assert b.model == "gpt-4o-mini"

    def test_registered_in_registry(self) -> None:
        _ensure_builtins_registered()
        assert "openai_vision" in list_backends()

    def test_skip_exists(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.openai_vision_backend import (
            OpenAIVisionBackend,
        )

        md_path = output_dir / "2401.00001v1.md"
        md_path.write_text("# Existing", encoding="utf-8")

        b = OpenAIVisionBackend(api_key="test_key")
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "skipped_exists"
        assert entry.converter_name == "openai_vision"

    def test_convert_import_error(self, pdf_file: Path, output_dir: Path) -> None:
        from research_pipeline.conversion.openai_vision_backend import (
            OpenAIVisionBackend,
        )

        b = OpenAIVisionBackend(api_key="test_key")
        # openai/fitz may not be installed in test env
        entry = b.convert(pdf_file, output_dir)
        assert entry.status == "failed"
        assert entry.error is not None


# ---------------------------------------------------------------------------
# Registry integration: all 8 backends registered
# ---------------------------------------------------------------------------


class TestAllBackendsRegistered:
    """Verify all 8 backends are registered after ensure call."""

    def test_all_eight_backends_present(self) -> None:
        _ensure_builtins_registered()
        names = list_backends()
        expected = [
            "datalab",
            "docling",
            "llamaparse",
            "marker",
            "mathpix",
            "mistral_ocr",
            "openai_vision",
            "pymupdf4llm",
        ]
        for name in expected:
            assert name in names, f"Backend {name!r} not registered"
        assert len(names) >= len(expected)

    def test_online_backends_require_credentials(self) -> None:
        """All online backends raise ValueError without credentials."""
        _ensure_builtins_registered()
        online = ["mathpix", "datalab", "llamaparse", "mistral_ocr", "openai_vision"]
        for name in online:
            with pytest.raises((ValueError, TypeError)):
                get_backend(name)
