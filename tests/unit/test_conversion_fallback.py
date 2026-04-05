"""Unit tests for FallbackConverter and multi-account configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_pipeline.config.models import (
    ConversionConfig,
    DatalabAccount,
    DatalabConfig,
    LlamaParseAccount,
    LlamaParseConfig,
    MathpixAccount,
    MathpixConfig,
    MistralOcrAccount,
    MistralOcrConfig,
    OpenAIVisionAccount,
    OpenAIVisionConfig,
)
from research_pipeline.conversion.base import ConverterBackend
from research_pipeline.conversion.fallback import (
    FallbackConverter,
    _is_quota_error,
)
from research_pipeline.infra.clock import utc_now
from research_pipeline.infra.hashing import sha256_str
from research_pipeline.models.conversion import ConvertManifestEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubBackend(ConverterBackend):
    """Stub backend for testing FallbackConverter."""

    def __init__(
        self,
        *,
        name: str = "stub",
        fail: bool = False,
        error_msg: str = "some error",
    ) -> None:
        self.name = name
        self.fail = fail
        self.error_msg = error_msg
        self.call_count = 0

    def fingerprint(self) -> str:
        return f"stub/{self.name}"

    def convert(
        self, pdf_path: Path, output_dir: Path, *, force: bool = False
    ) -> ConvertManifestEntry:
        self.call_count += 1
        if self.fail:
            return ConvertManifestEntry(
                arxiv_id="test",
                version="v1",
                pdf_path=str(pdf_path),
                pdf_sha256=sha256_str("fake"),
                markdown_path="",
                converter_name=self.name,
                converter_version="test",
                converter_config_hash="abc",
                converted_at=utc_now(),
                warnings=[self.error_msg],
                status="failed",
                error=self.error_msg,
            )
        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text("# Converted", encoding="utf-8")
        return ConvertManifestEntry(
            arxiv_id="test",
            version="v1",
            pdf_path=str(pdf_path),
            pdf_sha256=sha256_str("fake"),
            markdown_path=str(md_path),
            converter_name=self.name,
            converter_version="test",
            converter_config_hash="abc",
            converted_at=utc_now(),
            warnings=[],
            status="converted",
        )


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    pdf = tmp_path / "2401.00001v1.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return pdf


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# _is_quota_error
# ---------------------------------------------------------------------------


class TestIsQuotaError:
    def test_rate_limit_detected(self) -> None:
        assert _is_quota_error("HTTP 429 Too Many Requests")

    def test_quota_exceeded_detected(self) -> None:
        assert _is_quota_error("quota exceeded for this account")

    def test_insufficient_credits_detected(self) -> None:
        assert _is_quota_error("Insufficient credits remaining")

    def test_usage_limit_detected(self) -> None:
        assert _is_quota_error("usage limit reached")

    def test_normal_error_not_detected(self) -> None:
        assert not _is_quota_error("Connection timeout")

    def test_empty_string(self) -> None:
        assert not _is_quota_error("")

    def test_case_insensitive(self) -> None:
        assert _is_quota_error("RATE LIMIT exceeded")


# ---------------------------------------------------------------------------
# FallbackConverter
# ---------------------------------------------------------------------------


class TestFallbackConverter:
    def test_empty_backends_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            FallbackConverter([])

    def test_single_backend_success(self, pdf_file: Path, output_dir: Path) -> None:
        b = StubBackend(name="ok")
        fc = FallbackConverter([b])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "converted"
        assert b.call_count == 1

    def test_single_backend_failure(self, pdf_file: Path, output_dir: Path) -> None:
        b = StubBackend(name="fail", fail=True)
        fc = FallbackConverter([b])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "failed"

    def test_fallback_to_second_backend(self, pdf_file: Path, output_dir: Path) -> None:
        b1 = StubBackend(name="fail1", fail=True, error_msg="429 rate limit")
        b2 = StubBackend(name="ok2")
        fc = FallbackConverter([b1, b2])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "converted"
        assert result.converter_name == "ok2"
        assert b1.call_count == 1
        assert b2.call_count == 1

    def test_fallback_to_third_backend(self, pdf_file: Path, output_dir: Path) -> None:
        b1 = StubBackend(name="fail1", fail=True, error_msg="quota exceeded")
        b2 = StubBackend(name="fail2", fail=True, error_msg="connection error")
        b3 = StubBackend(name="ok3")
        fc = FallbackConverter([b1, b2, b3])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "converted"
        assert result.converter_name == "ok3"
        assert b1.call_count == 1
        assert b2.call_count == 1
        assert b3.call_count == 1

    def test_all_backends_fail(self, pdf_file: Path, output_dir: Path) -> None:
        b1 = StubBackend(name="fail1", fail=True, error_msg="err1")
        b2 = StubBackend(name="fail2", fail=True, error_msg="err2")
        fc = FallbackConverter([b1, b2])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "failed"
        # Returns the last error
        assert result.error == "err2"

    def test_fingerprint_includes_all_backends(self) -> None:
        b1 = StubBackend(name="a")
        b2 = StubBackend(name="b")
        fc = FallbackConverter([b1, b2])
        fp = fc.fingerprint()
        assert fp == "fallback(stub/a,stub/b)"

    def test_non_quota_error_also_falls_through(
        self, pdf_file: Path, output_dir: Path
    ) -> None:
        b1 = StubBackend(name="fail1", fail=True, error_msg="timeout error")
        b2 = StubBackend(name="ok2")
        fc = FallbackConverter([b1, b2])
        result = fc.convert(pdf_file, output_dir)
        assert result.status == "converted"


# ---------------------------------------------------------------------------
# Multi-account config models
# ---------------------------------------------------------------------------


class TestMultiAccountConfig:
    def test_mathpix_single_account_backward_compat(self) -> None:
        cfg = MathpixConfig(app_id="id1", app_key="key1")
        assert cfg.app_id == "id1"
        assert cfg.app_key == "key1"
        assert cfg.accounts == []

    def test_mathpix_multi_account(self) -> None:
        cfg = MathpixConfig(
            accounts=[
                MathpixAccount(app_id="id1", app_key="key1"),
                MathpixAccount(app_id="id2", app_key="key2"),
            ]
        )
        assert len(cfg.accounts) == 2
        assert cfg.accounts[0].app_id == "id1"
        assert cfg.accounts[1].app_id == "id2"

    def test_datalab_multi_account(self) -> None:
        cfg = DatalabConfig(
            accounts=[
                DatalabAccount(api_key="k1", mode="fast"),
                DatalabAccount(api_key="k2", mode="accurate"),
            ]
        )
        assert len(cfg.accounts) == 2
        assert cfg.accounts[0].mode == "fast"

    def test_datalab_account_default_mode(self) -> None:
        acct = DatalabAccount(api_key="k1")
        assert acct.mode == "balanced"

    def test_llamaparse_multi_account(self) -> None:
        cfg = LlamaParseConfig(
            accounts=[
                LlamaParseAccount(api_key="k1"),
                LlamaParseAccount(api_key="k2"),
            ]
        )
        assert len(cfg.accounts) == 2

    def test_llamaparse_tier_default(self) -> None:
        cfg = LlamaParseConfig()
        assert cfg.tier == "agentic"

    def test_llamaparse_tier_custom(self) -> None:
        cfg = LlamaParseConfig(tier="cost-effective")
        assert cfg.tier == "cost-effective"

    def test_llamaparse_account_tier_default(self) -> None:
        acct = LlamaParseAccount(api_key="k1")
        assert acct.tier == "agentic"

    def test_llamaparse_account_tier_override(self) -> None:
        acct = LlamaParseAccount(api_key="k1", tier="fast")
        assert acct.tier == "fast"

    def test_mistral_ocr_multi_account(self) -> None:
        cfg = MistralOcrConfig(
            accounts=[
                MistralOcrAccount(api_key="k1"),
                MistralOcrAccount(api_key="k2", model="custom-model"),
            ]
        )
        assert len(cfg.accounts) == 2
        assert cfg.accounts[0].model == "mistral-ocr-latest"
        assert cfg.accounts[1].model == "custom-model"

    def test_openai_vision_multi_account(self) -> None:
        cfg = OpenAIVisionConfig(
            accounts=[
                OpenAIVisionAccount(api_key="k1"),
                OpenAIVisionAccount(api_key="k2", model="gpt-4o-mini"),
            ]
        )
        assert len(cfg.accounts) == 2
        assert cfg.accounts[0].model == "gpt-4o"
        assert cfg.accounts[1].model == "gpt-4o-mini"

    def test_conversion_config_fallback_backends_default(self) -> None:
        cfg = ConversionConfig()
        assert cfg.fallback_backends == []

    def test_conversion_config_with_fallback(self) -> None:
        cfg = ConversionConfig(
            backend="mathpix",
            fallback_backends=["datalab", "mistral_ocr"],
        )
        assert cfg.backend == "mathpix"
        assert cfg.fallback_backends == ["datalab", "mistral_ocr"]


# ---------------------------------------------------------------------------
# _backend_kwargs_list
# ---------------------------------------------------------------------------


class TestBackendKwargsList:
    def test_single_account_mathpix(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.mathpix = MathpixConfig(app_id="id1", app_key="key1")
        result = _backend_kwargs_list("mathpix", config)
        assert result == [{"app_id": "id1", "app_key": "key1"}]

    def test_multi_account_mathpix(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.mathpix = MathpixConfig(
            accounts=[
                MathpixAccount(app_id="id1", app_key="key1"),
                MathpixAccount(app_id="id2", app_key="key2"),
            ]
        )
        result = _backend_kwargs_list("mathpix", config)
        assert len(result) == 2
        assert result[0] == {"app_id": "id1", "app_key": "key1"}
        assert result[1] == {"app_id": "id2", "app_key": "key2"}

    def test_multi_account_datalab(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.datalab = DatalabConfig(
            accounts=[
                DatalabAccount(api_key="k1", mode="fast"),
                DatalabAccount(api_key="k2", mode="accurate"),
            ]
        )
        result = _backend_kwargs_list("datalab", config)
        assert len(result) == 2
        assert result[0] == {"api_key": "k1", "mode": "fast"}
        assert result[1] == {"api_key": "k2", "mode": "accurate"}

    def test_multi_account_llamaparse(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.llamaparse = LlamaParseConfig(
            accounts=[
                LlamaParseAccount(api_key="k1"),
                LlamaParseAccount(api_key="k2"),
            ]
        )
        result = _backend_kwargs_list("llamaparse", config)
        assert len(result) == 2
        assert result[0] == {"api_key": "k1", "tier": "agentic"}
        assert result[1] == {"api_key": "k2", "tier": "agentic"}

    def test_multi_account_llamaparse_with_tier(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.llamaparse = LlamaParseConfig(
            accounts=[
                LlamaParseAccount(api_key="k1", tier="fast"),
                LlamaParseAccount(api_key="k2", tier="agentic-plus"),
            ]
        )
        result = _backend_kwargs_list("llamaparse", config)
        assert len(result) == 2
        assert result[0] == {"api_key": "k1", "tier": "fast"}
        assert result[1] == {"api_key": "k2", "tier": "agentic-plus"}

    def test_multi_account_mistral_ocr(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.mistral_ocr = MistralOcrConfig(
            accounts=[
                MistralOcrAccount(api_key="k1"),
                MistralOcrAccount(api_key="k2", model="custom"),
            ]
        )
        result = _backend_kwargs_list("mistral_ocr", config)
        assert len(result) == 2
        assert result[0]["model"] == "mistral-ocr-latest"
        assert result[1]["model"] == "custom"

    def test_multi_account_openai_vision(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        config.conversion.openai_vision = OpenAIVisionConfig(
            accounts=[
                OpenAIVisionAccount(api_key="k1"),
                OpenAIVisionAccount(api_key="k2", model="gpt-4o-mini"),
            ]
        )
        result = _backend_kwargs_list("openai_vision", config)
        assert len(result) == 2
        assert result[1]["model"] == "gpt-4o-mini"

    def test_local_backend_returns_single(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        result = _backend_kwargs_list("pymupdf4llm", config)
        assert result == [{}]

    def test_docling_returns_single(self) -> None:
        from research_pipeline.cli.cmd_convert import _backend_kwargs_list
        from research_pipeline.config.models import PipelineConfig

        config = PipelineConfig()
        result = _backend_kwargs_list("docling", config)
        assert result == [{"timeout_seconds": 300}]
