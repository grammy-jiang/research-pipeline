"""Prompt/response logging for LLM audit trail."""

import json
import logging
from pathlib import Path

from arxiv_paper_pipeline.infra.clock import utc_now
from arxiv_paper_pipeline.infra.hashing import sha256_str
from arxiv_paper_pipeline.models.manifest import LLMCallRecord

logger = logging.getLogger(__name__)


def log_llm_call(
    call_id: str,
    provider: str,
    model: str,
    prompt_version: str,
    input_payload: str,
    output_payload: str,
    duration_ms: int,
    token_usage: dict[str, int] | None = None,
    log_dir: Path | None = None,
) -> LLMCallRecord:
    """Create an LLM call audit record and optionally persist it.

    Args:
        call_id: Unique call identifier.
        provider: LLM provider name.
        model: Model identifier.
        prompt_version: Prompt template version.
        input_payload: Serialized input.
        output_payload: Serialized output.
        duration_ms: Call duration in milliseconds.
        token_usage: Token usage stats.
        log_dir: If set, write the full call record to this directory.

    Returns:
        LLMCallRecord for manifest inclusion.
    """
    record = LLMCallRecord(
        call_id=call_id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        input_hash=sha256_str(input_payload),
        output_hash=sha256_str(output_payload),
        token_usage=token_usage or {},
        called_at=utc_now(),
        duration_ms=duration_ms,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"llm_call_{call_id}.json"
        log_data = {
            "record": record.model_dump(mode="json"),
            "input": input_payload,
            "output": output_payload,
        }
        log_file.write_text(
            json.dumps(log_data, indent=2, default=str), encoding="utf-8"
        )
        logger.debug("Logged LLM call to %s", log_file)

    return record
