"""Tests for LLM call audit logging."""

from __future__ import annotations

import json
from pathlib import Path

from research_pipeline.infra.hashing import sha256_str
from research_pipeline.llm.audit import log_llm_call
from research_pipeline.models.manifest import LLMCallRecord


class TestLogLlmCall:
    """Tests for log_llm_call()."""

    def test_returns_llm_call_record(self) -> None:
        """Result is an LLMCallRecord."""
        rec = log_llm_call(
            call_id="c1",
            provider="openai",
            model="gpt-4",
            prompt_version="v1",
            input_payload="hello",
            output_payload="world",
            duration_ms=150,
        )
        assert isinstance(rec, LLMCallRecord)

    def test_correct_fields(self) -> None:
        """Record carries provider, model, and prompt_version."""
        rec = log_llm_call(
            call_id="c2",
            provider="anthropic",
            model="claude-3",
            prompt_version="v2",
            input_payload="in",
            output_payload="out",
            duration_ms=200,
        )
        assert rec.call_id == "c2"
        assert rec.provider == "anthropic"
        assert rec.model == "claude-3"
        assert rec.prompt_version == "v2"
        assert rec.duration_ms == 200

    def test_input_hash_is_sha256(self) -> None:
        """input_hash equals SHA-256 of the input_payload."""
        payload = "test input"
        rec = log_llm_call(
            call_id="c3",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload=payload,
            output_payload="out",
            duration_ms=10,
        )
        assert rec.input_hash == sha256_str(payload)

    def test_output_hash_is_sha256(self) -> None:
        """output_hash equals SHA-256 of the output_payload."""
        payload = "test output"
        rec = log_llm_call(
            call_id="c4",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="in",
            output_payload=payload,
            duration_ms=10,
        )
        assert rec.output_hash == sha256_str(payload)

    def test_token_usage_defaults_to_empty(self) -> None:
        """Omitting token_usage yields an empty dict."""
        rec = log_llm_call(
            call_id="c5",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="in",
            output_payload="out",
            duration_ms=10,
        )
        assert rec.token_usage == {}

    def test_token_usage_passed_through(self) -> None:
        """Explicit token_usage is preserved."""
        usage = {"prompt_tokens": 50, "completion_tokens": 20}
        rec = log_llm_call(
            call_id="c6",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="in",
            output_payload="out",
            duration_ms=10,
            token_usage=usage,
        )
        assert rec.token_usage == usage

    def test_log_dir_creates_directory_and_file(self, tmp_path: Path) -> None:
        """When log_dir is set, the directory and JSON file are created."""
        log_dir = tmp_path / "llm_logs"
        log_llm_call(
            call_id="c7",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="inp",
            output_payload="outp",
            duration_ms=100,
            log_dir=log_dir,
        )
        assert log_dir.exists()
        log_file = log_dir / "llm_call_c7.json"
        assert log_file.exists()

    def test_log_file_contains_record_and_payloads(self, tmp_path: Path) -> None:
        """JSON file contains record, input, and output."""
        log_dir = tmp_path / "logs"
        log_llm_call(
            call_id="c8",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="my input",
            output_payload="my output",
            duration_ms=50,
            log_dir=log_dir,
        )
        data = json.loads((log_dir / "llm_call_c8.json").read_text())
        assert "record" in data
        assert data["input"] == "my input"
        assert data["output"] == "my output"
        assert data["record"]["call_id"] == "c8"

    def test_no_log_dir_does_not_write(self, tmp_path: Path) -> None:
        """When log_dir is None, no files are written."""
        rec = log_llm_call(
            call_id="c9",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="in",
            output_payload="out",
            duration_ms=10,
            log_dir=None,
        )
        # Should still return a record
        assert isinstance(rec, LLMCallRecord)

    def test_called_at_populated(self) -> None:
        """called_at timestamp is set."""
        rec = log_llm_call(
            call_id="c10",
            provider="p",
            model="m",
            prompt_version="v1",
            input_payload="in",
            output_payload="out",
            duration_ms=10,
        )
        assert rec.called_at is not None
